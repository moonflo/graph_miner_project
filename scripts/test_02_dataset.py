"""Acceptance test 02: local ogbl-collab valid split integrity."""
from __future__ import annotations
import argparse
from _acceptance_common import add_common_args, check, finish, setup
from src.graph.dataset_registry import require_supported_dataset
from src.graph.ogb_split_loader import load_ogb_split
from src.graph.candidates import candidates_from_split, edges_to_node_pairs

def main() -> int:
 p=argparse.ArgumentParser(description=__doc__); add_common_args(p); a=p.parse_args(); log,out,d=setup("ogbl-collab数据完整性检查","02_dataset.log",a.output_dir); d["inputs"]={"raw_root":a.raw_root,"dataset":"ogbl_collab","split":"valid"}
 try:
  name=require_supported_dataset("ogbl_collab"); c=candidates_from_split(load_ogb_split(name,a.raw_root),"valid"); pos=edges_to_node_pairs(c.positive_edges); neg=edges_to_node_pairs(c.negative_edges)
  actual={"positive_count":len(pos),"negative_count":len(neg),"positive_shape":list(c.positive_edges.shape),"negative_shape":list(c.negative_edges.shape) if c.negative_edges is not None else None}; d["actual"]=actual
  check(d,"验证集正关系",len(pos),60084,len(pos)==60084); check(d,"验证集候选负关系",len(neg),100000,len(neg)==100000); check(d,"关系结构",actual,"每条关系两个节点",all(len(x)==2 for x in [*pos[:3],*neg[:3]]))
  ok=all(x["passed"] for x in d["checks"]); return finish(log,out,d,"02_dataset.json","PASS" if ok else "FAIL","通过" if ok else "未通过")
 except Exception as e: d["errors"].append(f"{type(e).__name__}: {e}"); return finish(log,out,d,"02_dataset.json","ERROR","执行异常")
if __name__=="__main__": raise SystemExit(main())
