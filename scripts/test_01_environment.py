"""Acceptance test 01: environment and permissions."""
from __future__ import annotations
import argparse, importlib, os, tempfile
from pathlib import Path
from _acceptance_common import add_common_args, check, env_summary, finish, setup

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__); add_common_args(p); a=p.parse_args()
    log,out,d=setup("环境部署与运行条件检查","01_environment.log",a.output_dir); d["inputs"]={"raw_root":a.raw_root,"output_dir":a.output_dir}; d["environment"]=env_summary()
    try:
        py_ok=os.sys.version_info >= (3,10); check(d,"Python版本",platform_version:=env_summary()["python"],">=3.10",py_ok)
        modules=["networkx","ogb","numpy","scipy","torch","src.algorithms.evaluation","src.algorithms.link_prediction","src.algorithms.scoring","src.graph.graph_factory","src.graph.ogb_split_loader"]
        imports={}
        for name in modules:
            try: imports[name]=getattr(importlib.import_module(name),"__version__","imported")
            except Exception as e: imports[name]=f"ERROR: {type(e).__name__}: {e}"; d["errors"].append(imports[name])
        d["imports"]=imports; check(d,"项目及第三方依赖导入",imports,"全部成功",not d["errors"])
        with tempfile.NamedTemporaryFile(dir=out, delete=True): pass
        check(d,"输出目录可写且临时文件可创建删除",True,True,True)
        return finish(log,out,d,"01_environment.json","PASS" if py_ok and not d["errors"] else "FAIL","通过" if py_ok and not d["errors"] else "未通过")
    except Exception as e: d["errors"].append(f"{type(e).__name__}: {e}"); return finish(log,out,d,"01_environment.json","ERROR","执行异常")
if __name__=="__main__": raise SystemExit(main())
