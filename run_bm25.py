"""V1-compatible command routed to auditable V2 runner; original source in legacy/v1."""
import argparse
from lab.runner import run

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True)
    p.add_argument('--model',default='cross-encoder/ms-marco-MiniLM-L-6-v2')
    p.add_argument('--top-k',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--rerank-depth',type=int,default=100)
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    p.add_argument('--download',action='store_true')
    args=p.parse_args()
    config=dict(dataset=args.dataset,retriever='bm25',reranker=None,top_k=args.top_k,
                batch_size=args.batch_size,candidate_k=args.rerank_depth,device=args.device,download=args.download)
    folder=run(config)
    import json
    if json.loads((folder/"status.json").read_text())["status"]!="VERIFIED":
        raise SystemExit(1)

if __name__=="__main__":
    main()
