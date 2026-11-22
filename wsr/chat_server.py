"""Local playground for the unmodified pretrained Mamba2 model."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time

import torch
from tokenizers import Tokenizer
from wsr.pretrained_model import PretrainedMamba2

PAGE = '''<!doctype html><meta charset="utf-8"><title>Mamba2 130M 试用</title>
<style>body{max-width:850px;margin:40px auto;font:16px system-ui;padding:0 20px;background:#f5f6f8}textarea,pre{box-sizing:border-box;width:100%;padding:16px;border:1px solid #ccc;border-radius:8px;background:white}textarea{height:170px;font:inherit}pre{white-space:pre-wrap;min-height:120px}button,select,input{padding:8px;margin:6px}small{color:#555}</style>
<h1>Mamba2 130M 试用</h1><p>官方原始预训练模型，未经指令微调。对话模式使用 User / Assistant 文本提示，模型可能答非所问或继续编写双方台词。建议先试英文。</p>
<label>模式 <select id="mode"><option value="chat">对话</option><option value="completion">文本续写</option></select></label>
<label>生成 token 数 <input id="count" type="number" value="100" min="1" max="256"></label>
<label>温度 <input id="temperature" type="number" value="0.8" min="0" max="2" step="0.1"></label>
<pre id="history">尚无对话</pre><textarea id="prompt">Hello! What is your name?</textarea><br>
<button id="send" onclick="generate()">生成</button><button onclick="history=[];document.getElementById('history').textContent='尚无对话'">清空对话</button><small id="status"></small><pre id="output">生成结果显示在这里</pre>
<script>
let history=[];
async function generate(){const b=document.getElementById('send'),s=document.getElementById('status');b.disabled=true;s.textContent='正在生成…';
const mode=document.getElementById('mode').value,prompt=document.getElementById('prompt').value;
try{const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt,mode,history,max_new_tokens:Number(document.getElementById('count').value),temperature:Number(document.getElementById('temperature').value)})});const d=await r.json();if(!r.ok)throw Error(d.error);document.getElementById('output').textContent=d.text;s.textContent=`${d.tokens} tokens · ${d.seconds.toFixed(1)} 秒`;if(mode==='chat'){history.push({user:prompt,assistant:d.text});document.getElementById('history').textContent=history.map(x=>'你：'+x.user+'\n模型：'+x.assistant).join('\n\n');}}
catch(e){s.textContent=e.message}finally{b.disabled=false}}
</script>'''
# Preserve JavaScript newline escapes inside the Python string.
PAGE = PAGE.replace("+'\n模型：'", "+'\\n模型：'").replace(".join('\n\n')", ".join('\\n\\n')")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=7860)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args()
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parents[1]/'artifacts/pretrained-v1/model'
    tokenizer=Tokenizer.from_file(str(root/'tokenizer.json'))
    model=PretrainedMamba2.load(root,device=args.device)
    lock=threading.Lock()
    eos=tokenizer.token_to_id('<|endoftext|>')

    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,data,kind='application/json'):
            body=data.encode() if isinstance(data,str) else json.dumps(data,ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type',kind+'; charset=utf-8')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path=='/': self.reply(200,PAGE,'text/html')
            elif self.path=='/health': self.reply(200,{'ready':True,'model':'state-spaces/mamba2-130m','device':args.device})
            else: self.reply(404,{'error':'Not found'})

        def do_POST(self):
            if self.path!='/api/generate': return self.reply(404,{'error':'Not found'})
            acquired=False
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=65536: raise ValueError('请求大小必须在 1–65536 字节之间')
                data=json.loads(self.rfile.read(size))
                prompt=data['prompt']
                if not isinstance(prompt,str) or not prompt.strip(): raise ValueError('请输入文本')
                count=int(data.get('max_new_tokens',100))
                temperature=float(data.get('temperature',0.8))
                if not 1<=count<=256 or not 0<=temperature<=2: raise ValueError('生成长度或温度超出范围')
                mode=data.get('mode','chat')
                if mode not in ('chat','completion'): raise ValueError('未知模式')
                if mode=='chat':
                    history=data.get('history',[])
                    if not isinstance(history,list) or len(history)>100: raise ValueError('对话记录过长')
                    prefix=''.join('User: '+x['user']+'\nAssistant: '+x['assistant']+'\n' for x in history)
                    prompt=prefix+'User: '+prompt+'\nAssistant:'
                ids=tokenizer.encode(prompt).ids
                if len(ids)>2048: raise ValueError('输入超过 2048 tokens，请清空对话或缩短文本')
                acquired=lock.acquire(blocking=False)
                if not acquired: return self.reply(503,{'error':'模型正在生成，请稍后重试'})
                start=time.monotonic()
                output=[]
                with torch.inference_mode():
                    cache=None
                    for offset in range(0,len(ids),128):
                        logits,cache=model.run(torch.tensor([ids[offset:offset+128]],device=args.device),cache,num_last_tokens=1)
                    for _ in range(count):
                        scores=logits[0,-1,:tokenizer.get_vocab_size()].float()
                        if temperature==0: token=int(scores.argmax())
                        else:
                            values,indices=torch.topk(scores/temperature,40)
                            token=int(indices[torch.multinomial(values.softmax(-1),1)])
                        if token==eos: break
                        output.append(token)
                        if mode=='chat' and '\nUser:' in tokenizer.decode(output): break
                        if len(output)<count:
                            logits,cache=model.run(torch.tensor([[token]],device=args.device),cache,num_last_tokens=1)
                result=tokenizer.decode(output)
                if mode=='chat': result=result.split('\nUser:',1)[0].strip()
                self.reply(200,{'text':result,'tokens':len(output),'seconds':time.monotonic()-start})
            except (ValueError,KeyError,TypeError) as exc: self.reply(400,{'error':str(exc)})
            except Exception as exc:
                self.log_error('%s',exc)
                self.reply(500,{'error':'推理失败，请查看服务日志'})
            finally:
                if acquired: lock.release()

    server=ThreadingHTTPServer((args.host,args.port),Handler)
    print(f'Ready: http://{args.host}:{args.port} ({args.device})',flush=True)
    server.serve_forever()

if __name__=='__main__': main()
