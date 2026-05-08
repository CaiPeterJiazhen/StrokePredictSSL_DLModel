# StrokePredictSSL-DLModel 开发说明

本仓库只提交代码、配置、文档和去标识化输出。原始 Excel、`.set`、`.fdt` 和运行输出均不提交。

Phase 0+1 常用命令：

```bash
python scripts/00_validate_environment.py
python scripts/01_build_cohort.py --config configs/project.yaml
python -m pytest tests -q
```

Windows ACL 受限目录下可使用显式临时目录并关闭 pytest cache：

```bash
python -m pytest tests --basetemp=F:\CJZProjectFile\StrokePredictSSL-DLModel\.codex_pytest_tmp -p no:cacheprovider
```
