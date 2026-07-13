"""Acceptance test 04: legacy global-pool Hits@50."""
from __future__ import annotations
import argparse, json
from _acceptance_common import add_common_args, check, finish, setup
from src.algorithms.evaluation import evaluate_ogb_style
from src.algorithms.scoring import score_multiple_methods_for_dataset

NOTE="本测试采用legacy全局负样本池评估方式。验证集正关系分别与同一个包含100,000条负关系的全局负样本池比较。--limit-neg-per-pos 100是负样本预算乘数，实际数量受官方可用负样本限制。本测试不是为每条正关系分别生成100条负关系，也不是在101个候选关系中计算Hits@50。"
def main() -> int:
 p=argparse.ArgumentParser(description=__doc__); add_common_args(p); a=p.parse_args(); log,out,d=setup("图拓扑学习关系识别准确率测试","04_link_prediction.log",a.output_dir); d["inputs"]={"raw_root":a.raw_root,"dataset":"ogbl_collab","split":"valid","method":"time_decay_common_neighbors","decay":0.8,"full_positive_split":True,"limit_neg_per_pos":100,"limit_train_edges":None}; log.info(NOTE)
 try:
  r=score_multiple_methods_for_dataset("ogbl_collab",["time_decay_common_neighbors"],"valid",raw_root=a.raw_root,limit_pos=None,limit_neg_per_pos=100,limit_train_edges=None,full_positive_split=True,decay=0.8)[0]; metric=evaluate_ogb_style("ogbl_collab",r)["Hits@50"]; g=r.graph_metadata; d.update({"evaluation_mode":"legacy_global_negative_pool","positive_count":len(r.pos_scores),"requested_negative_count":r.requested_negative_count,"available_negative_count":r.available_negative_count,"used_negative_count":r.negative_count,"negative_truncated":r.negative_truncated,"graph":g,"metric":{"name":"Hits@50","value":metric,"threshold":0.60,"passed":metric>=0.60},"negative_pool_note":NOTE});
  log.info("数据集=ogbl_collab split=valid method=time_decay_common_neighbors decay=0.8")
  log.info("正关系=%s 请求负关系=%s 可用负关系=%s 实际使用负关系=%s 截断=%s", f"{len(r.pos_scores):,}", f"{r.requested_negative_count:,}", f"{r.available_negative_count:,}", f"{r.negative_count:,}", r.negative_truncated)
  log.info("完整训练图=%s节点/%s关系 Hits@50=%.6f 阈值=0.600000", f"{g.get('num_nodes'):,}", f"{g.get('num_edges'):,}", metric)
  for n,x,e in [("正关系数量",len(r.pos_scores),60084),("可用负关系数量",r.available_negative_count,100000),("实际使用负关系数量",r.negative_count,100000),("图节点数量",g.get("num_nodes"),235868),("图关系数量",g.get("num_edges"),967632),("Hits@50",metric,0.60)]: check(d,n,x,e,(x==e if n!="Hits@50" else x>=e))
  ok=all(x["passed"] for x in d["checks"]); status="PASS" if ok else "FAIL"; conclusion="通过" if ok else "未通过"; result=finish(log,out,d,"04_link_prediction.json",status,conclusion)
  (out/"04_link_prediction.md").write_text(f"# 图拓扑学习关系识别准确率测试\n\n- 状态：**{status}**\n- Hits@50：{metric:.6f}\n- 阈值：0.600000\n- 正关系：{len(r.pos_scores):,}\n- 负关系：{r.negative_count:,}（请求 {r.requested_negative_count:,}，可用 {r.available_negative_count:,}，截断：{r.negative_truncated}）\n- 算法：time_decay_common_neighbors，decay=0.8\n\n{NOTE}\n\n排名定义：1 + 得分大于或等于正关系得分的全局负关系数量；排名不大于50计为命中。\n",encoding="utf-8"); return result
 except Exception as e: d["errors"].append(f"{type(e).__name__}: {e}"); return finish(log,out,d,"04_link_prediction.json","ERROR","执行异常")
if __name__=="__main__": raise SystemExit(main())
