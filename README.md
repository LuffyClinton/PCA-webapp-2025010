# PCA分析工具：Web应用版

## 项目简介
基于Flask开发的零代码PCA（主成分分析）工具，支持Excel文件上传，自动完成数据预处理、降维分析、可视化生成与报告导出，适合非技术用户快速开展维度 reduction 任务。


## 核心功能
- **零代码操作**：上传Excel即可生成完整分析结果
- **自动化分析**：
  - 自动计算最优主成分数（默认90%方差阈值）
  - 支持标准化/最小最大/鲁棒3种数据缩放
  - Hotelling's T²异常值检测
- **可视化输出**：
  - PCA综合仪表盘（样本分布+异常值+方差曲线）
  - 双标图（Biplot）、特征相关性热图、特征重要性排序
- **报告导出**：多工作表Excel（含原始数据、主成分矩阵、异常结果等）


## 快速开始
### 1. 环境要求
- Python 3.7+
- 依赖：`flask pandas numpy matplotlib scipy openpyxl`

### 2. 安装依赖
```bash
pip install flask pandas numpy matplotlib scipy openpyxl
```

### 3. 启动应用
```bash
# 进入项目根目录
cd pca_webapp
# 运行主程序
python app.py
# 浏览器访问：http://127.0.0.1:5000
```


## 使用指南
### 1. 数据准备
- 格式：Excel（.xlsx/.xls）
- 结构：第一行表头（特征名），后续为纯数值（无空值/文本）

### 2. 操作步骤
1. 上传Excel文件 → 点击"开始分析"
2. 查看结果页（含分析摘要+可视化图表）
3. 下载Excel报告 → 完成分析


## 项目结构
```
pca_webapp/
├── app.py               # 主程序（Flask后端）
├── templates/
│   ├── index.html       # 上传页面
│   └── results.html     # 结果展示页面
├── static/
│   ├── css/
│   │   └── style.css    # 简单样式
│   └── uploads/         # 临时存储上传的文件
│   └── results/         # 存储生成的结果（图表+Excel）
```


## 许可证
MIT 开源许可证，可自由使用、修改与分发。

## 联系开发者
- 邮箱：gonghd3@mail2.sysu.edu.cn
