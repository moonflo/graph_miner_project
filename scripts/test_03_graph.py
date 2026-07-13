"""Acceptance test 03: complete official training graph."""
from __future__ import annotations
import argparse
from _acceptance_common import add_common_args, check, finish, setup
from src.graph.ogb_split_loader import load_ogb_split
from src.graph.graph_factory import build_networkx_graph_from_train_split
from src.algorithms.link_prediction import to_simple_undirected_for_topology

def main() -> int:
 p=argparse.ArgumentParser(description=__doc__); add_common_args(p); a=p.parse_args(); log,out,d=setup("完整训练图构建与属性检查","03_graph.log",a.output_dir); d["inputs"]={"raw_root":a.raw_root,"dataset":"ogbl_collab","limit_edges":None,"include_isolated_nodes":True}
 try:
  g=build_networkx_graph_from_train_split("ogbl_collab",raw_root=a.raw_root,limit_edges=None,include_isolated_nodes=True,split_data=load_ogb_split("ogbl_collab",a.raw_root)); t=to_simple_undirected_for_topology(g,include_isolated_nodes=False); m={"num_nodes":g.number_of_nodes(),"num_edges":g.number_of_edges(),"has_edge_weight":g.graph.get("has_edge_weight"),"has_edge_year":g.graph.get("has_edge_year"),"max_train_year":g.graph.get("max_train_year"),"topology_num_nodes":t.number_of_nodes(),"topology_num_edges":t.number_of_edges()}; d["graph"]=m
  for n,x,e in [("节点数量",m["num_nodes"],235868),("关系数量",m["num_edges"],967632),("边权重",m["has_edge_weight"],True),("边年份",m["has_edge_year"],True),("最大训练年份",m["max_train_year"],2017)]: check(d,n,x,e,x==e)
  ok=all(x["passed"] for x in d["checks"]); return finish(log,out,d,"03_graph.json","PASS" if ok else "FAIL","通过" if ok else "未通过")
 except Exception as e: d["errors"].append(f"{type(e).__name__}: {e}"); return finish(log,out,d,"03_graph.json","ERROR","执行异常")
if __name__=="__main__": raise SystemExit(main())
