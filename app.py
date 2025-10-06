import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，确保图表可保存
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import uuid  # 生成唯一ID，避免文件重名

# 初始化Flask应用
app = Flask(__name__)

# 配置
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}  # 允许的文件类型

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# 创建必要的文件夹
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


# PCA类（复用你的核心代码，略作修改适配Web环境）
class PCA:
    def __init__(self, n_components=None, scaling='standard', random_state=None):
        self.n_components = n_components
        self.scaling = scaling
        self.random_state = random_state
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.scale_ = None
        self.n_samples_ = None
        self.n_features_ = None
        self.feature_names_ = None

    def _scale_data(self, X):
        if self.scaling is None:
            return X
        
        if self.mean_ is None:  # 仅在fit时计算，transform复用
            self.mean_ = np.mean(X, axis=0)
            X_centered = X - self.mean_
            
            if self.scaling == 'standard':
                self.scale_ = np.std(X, axis=0, ddof=1)
            elif self.scaling == 'minmax':
                self.scale_ = np.max(X, axis=0) - np.min(X, axis=0)
            elif self.scaling == 'robust':
                q75, q25 = np.percentile(X, [75, 25], axis=0)
                self.scale_ = q75 - q25
            else:
                raise ValueError(f"Unknown scaling method: {self.scaling}")
        else:
            X_centered = X - self.mean_
        
        self.scale_[self.scale_ == 0] = 1e-8  # 避免除0
        return X_centered / self.scale_

    def fit(self, X, feature_names=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.n_samples_, self.n_features_ = X.shape
        
        self.feature_names_ = feature_names if feature_names is not None else [f'Feature_{i+1}' for i in range(self.n_features_)]
        
        X_scaled = self._scale_data(X)
        cov_matrix = np.cov(X_scaled, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        if self.n_components is None:
            self.n_components = min(self.n_samples_, self.n_features_)
        
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(eigenvalues)
        self.singular_values_ = np.sqrt(self.explained_variance_ * (self.n_samples_ - 1))
        
        return self
    
    def transform(self, X):
        X_scaled = self._scale_data(X)
        return np.dot(X_scaled, self.components_.T)
    
    def fit_transform(self, X, feature_names=None):
        self.fit(X, feature_names=feature_names)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        X_reconstructed = np.dot(X_transformed, self.components_)
        if self.scaling is not None:
            X_reconstructed = X_reconstructed * self.scale_
        return X_reconstructed + self.mean_
    
    def reconstruction_error(self, X):
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        return np.mean(np.sum((X - X_reconstructed) **2, axis=1))
    
    def get_feature_importance(self):
        weighted_loadings = np.abs(self.components_.T) * np.sqrt(self.explained_variance_)
        return np.sum(weighted_loadings, axis=1)
    
    def hotelling_t2(self, X):
        X_transformed = self.transform(X)
        t2 = np.zeros(X.shape[0])
        for i in range(self.n_components):
            t2 += (X_transformed[:, i]** 2) / self.explained_variance_[i]
        return t2
    
    def find_outliers(self, X, alpha=0.05):
        t2 = self.hotelling_t2(X)
        df1 = self.n_components
        df2 = self.n_samples_ - self.n_components
        factor = df1 * (self.n_samples_ **2 - 1) / (self.n_samples_ * df2)
        critical_value = stats.f.ppf(1 - alpha, df1, df2) * factor
        return t2 > critical_value
    
    def plot_explained_variance(self, cumulative=True, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 6))
        n_components = len(self.explained_variance_ratio_)
        ind = np.arange(n_components)
        
        if cumulative:
            cumsum = np.cumsum(self.explained_variance_ratio_)
            ax.bar(ind, self.explained_variance_ratio_, alpha=0.5)
            ax.step(ind, cumsum, where='mid', label='Cumulative')
            ax.axhline(y=0.9, color='r', linestyle='-', label='90% Explained')
            ax.set_ylabel('Cumulative Explained Variance Ratio')
            ax.legend()
        else:
            ax.bar(ind, self.explained_variance_ratio_)
            ax.set_ylabel('Explained Variance Ratio')
        
        ax.set_xlabel('Principal Components')
        ax.set_title('Explained Variance by Components')
        ax.set_xticks(ind)
        ax.set_xticklabels([f'PC{i+1}' for i in ind])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    
    def biplot(self, X, n_features=5, pc_x=0, pc_y=1, save_path=None):
        X_transformed = self.transform(X)
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(X_transformed[:, pc_x], X_transformed[:, pc_y], alpha=0.7, color='steelblue')
        
        feature_names = self.feature_names_
        loadings = self.components_[[pc_x, pc_y], :]
        loading_matrix = loadings.T
        loading_magnitude = np.sum(loading_matrix**2, axis=1)
        sorted_indices = np.argsort(loading_magnitude)[::-1][:n_features]
        
        scale = np.max(np.abs(X_transformed[:, [pc_x, pc_y]])) / np.max(np.abs(loadings))
        
        for i in sorted_indices:
            ax.arrow(0, 0, 
                     loading_matrix[i, 0] * scale * 0.8, 
                     loading_matrix[i, 1] * scale * 0.8, 
                     head_width=scale*0.05, head_length=scale*0.08, 
                     fc='red', ec='red')
            ax.text(loading_matrix[i, 0] * scale * 0.85, 
                    loading_matrix[i, 1] * scale * 0.85, 
                    feature_names[i], color='red')
        
        ax.set_xlabel(f'PC{pc_x+1} ({self.explained_variance_ratio_[pc_x]:.2%})')
        ax.set_ylabel(f'PC{pc_y+1} ({self.explained_variance_ratio_[pc_y]:.2%})')
        ax.set_title('PCA Biplot')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    
    def get_optimal_components(self, threshold=0.95):
        cumsum = np.cumsum(self.explained_variance_ratio_)
        return np.argmax(cumsum >= threshold) + 1

    def export_results_to_excel(self, X, output_path):
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_original = pd.DataFrame(X, columns=self.feature_names_)
            df_original.to_excel(writer, sheet_name='Original Data', index=False)
            
            X_scaled = self._scale_data(X)
            df_scaled = pd.DataFrame(X_scaled, columns=self.feature_names_)
            df_scaled.to_excel(writer, sheet_name='Scaled Data', index=False)
            
            df_components = pd.DataFrame(
                self.components_,
                index=[f'Component_{i+1}' for i in range(self.n_components)],
                columns=self.feature_names_
            )
            df_components.to_excel(writer, sheet_name='Component Matrix')
            
            df_variance = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(self.n_components)],
                'Explained Variance': self.explained_variance_,
                'Variance Ratio': self.explained_variance_ratio_,
                'Cumulative Ratio': np.cumsum(self.explained_variance_ratio_)
            })
            df_variance.to_excel(writer, sheet_name='Variance Explanation', index=False)
            
            X_transformed = self.transform(X)
            df_transformed = pd.DataFrame(
                X_transformed,
                columns=[f'Component_{i+1}' for i in range(self.n_components)]
            )
            df_transformed.to_excel(writer, sheet_name='Transformed Data', index=False)
            
            X_reconstructed = self.inverse_transform(X_transformed)
            df_reconstructed = pd.DataFrame(X_reconstructed, columns=self.feature_names_)
            df_reconstructed.to_excel(writer, sheet_name='Reconstructed Data', index=False)
            
            importance = self.get_feature_importance()
            df_importance = pd.DataFrame({
                'Feature': self.feature_names_,
                'Importance Score': importance
            }).sort_values(by='Importance Score', ascending=False)
            df_importance.to_excel(writer, sheet_name='Feature Importance', index=False)
            
            t2 = self.hotelling_t2(X)
            outliers = self.find_outliers(X)
            df_outliers = pd.DataFrame({
                'Sample Index': np.arange(len(X)),
                "Hotelling's T²": t2,
                'Is Outlier': ['Yes' if out else 'No' for out in outliers]
            })
            df_outliers.to_excel(writer, sheet_name='Outlier Detection', index=False)
        return output_path


# 辅助函数：检查文件是否合法
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 生成所有可视化图表
def generate_visualizations(pca, X, X_transformed, results_id):
    vis_paths = {}
    
    # 1. PCA Dashboard（4子图）
    dashboard_path = os.path.join(app.config['RESULTS_FOLDER'], f'{results_id}_pca_dashboard.png')
    plt.figure(figsize=(12, 10))
    
    # 子图1：前2维+Hotelling T²
    plt.subplot(2, 2, 1)
    t2_values = pca.hotelling_t2(X)
    scatter = plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=t2_values, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label="Hotelling's T²")
    plt.title('Distribution of First Two Components')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 子图2：3D可视化（若有）
    if pca.n_components >= 3:
        from mpl_toolkits.mplot3d import Axes3D
        ax = plt.subplot(2, 2, 2, projection='3d')
        ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2], c=t2_values, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label="Hotelling's T²")
        ax.set_title('3D Distribution of First Three Components')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
    else:
        plt.subplot(2, 2, 2)
        plt.text(0.5, 0.5, 'Not enough components for 3D plot', ha='center', va='center', transform=plt.gca().transAxes)
        plt.axis('off')
    
    # 子图3：异常值标记
    plt.subplot(2, 2, 3)
    outliers = pca.find_outliers(X)
    plt.scatter(X_transformed[~outliers, 0], X_transformed[~outliers, 1], marker='o', color='steelblue', alpha=0.8, label='Normal')
    plt.scatter(X_transformed[outliers, 0], X_transformed[outliers, 1], marker='*', color='crimson', s=100, alpha=0.8, label='Outliers')
    plt.title(f'PC1 vs PC2 (Outliers: {np.sum(outliers)})')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 子图4：累积方差
    plt.subplot(2, 2, 4)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(np.arange(1, len(cumsum)+1), cumsum, 'o-', linewidth=2)
    plt.axhline(y=0.9, color='red', linestyle='--', label='90% Threshold')
    plt.axvline(x=pca.n_components, color='green', linestyle='--', label=f'Optimal: {pca.n_components}')
    plt.title('Cumulative Variance Contribution')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Ratio')
    plt.xticks(np.arange(1, len(cumsum)+1, step=max(1, len(cumsum)//5)))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
    plt.close()
    vis_paths['dashboard'] = dashboard_path
    
    # 2. Biplot
    biplot_path = os.path.join(app.config['RESULTS_FOLDER'], f'{results_id}_pca_biplot.png')
    pca.biplot(X, n_features=min(pca.n_features_, 10), pc_x=0, pc_y=1, save_path=biplot_path)
    vis_paths['biplot'] = biplot_path
    
    # 3. 原始vs重构数据
    recon_path = os.path.join(app.config['RESULTS_FOLDER'], f'{results_id}_pca_reconstruction.png')
    X_reconstructed = pca.inverse_transform(X_transformed)
    reconstruction_error = pca.reconstruction_error(X)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label='Original', color='steelblue')
    plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], alpha=0.5, label='Reconstructed', color='orange')
    for i in range(min(50, X.shape[0])):
        plt.plot([X[i, 0], X_reconstructed[i, 0]], [X[i, 1], X_reconstructed[i, 1]], 'k-', alpha=0.1)
    plt.title(f'Original vs Reconstructed (MSE: {reconstruction_error:.4f})')
    plt.xlabel(pca.feature_names_[0])
    plt.ylabel(pca.feature_names_[1])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    sample_errors = np.sum((X - X_reconstructed) **2, axis=1)
    plt.bar(range(len(sample_errors)), sample_errors)
    plt.axhline(np.mean(sample_errors), color='red', linestyle='--', label=f'Mean: {np.mean(sample_errors):.4f}')
    plt.title('Sample Reconstruction Errors')
    plt.xlabel('Sample Index')
    plt.ylabel('Squared Error')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(recon_path, dpi=300, bbox_inches='tight')
    plt.close()
    vis_paths['reconstruction'] = recon_path
    
    # 4. 特征相关性矩阵
    corr_path = os.path.join(app.config['RESULTS_FOLDER'], f'{results_id}_feature_correlation.png')
    correlation_matrix = np.corrcoef(X.T)
    plt.figure(figsize=(10, 8))
    im = plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, label='Correlation Coefficient')
    plt.title('Variable Correlation Matrix')
    plt.xticks(np.arange(pca.n_features_), pca.feature_names_, rotation=45, ha='right', fontsize=8)
    plt.yticks(np.arange(pca.n_features_), pca.feature_names_, fontsize=8)
    for i in range(pca.n_features_):
        for j in range(pca.n_features_):
            plt.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                     ha='center', va='center',
                     color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black',
                     fontsize=7)
    plt.tight_layout()
    plt.savefig(corr_path, dpi=300, bbox_inches='tight')
    plt.close()
    vis_paths['correlation'] = corr_path
    
    # 5. 特征重要性
    imp_path = os.path.join(app.config['RESULTS_FOLDER'], f'{results_id}_variable_importance.png')
    importance = pca.get_feature_importance()
    sorted_idx = np.argsort(importance)[::-1]
    sorted_features = [pca.feature_names_[idx] for idx in sorted_idx]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importance)), importance[sorted_idx])
    plt.xticks(range(len(importance)), sorted_features, rotation=45, ha='right', fontsize=8)
    plt.title('Variable Importance (Based on PCA Loadings)')
    plt.xlabel('Variables (Sorted by Importance)')
    plt.ylabel('Importance Score')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(imp_path, dpi=300, bbox_inches='tight')
    plt.close()
    vis_paths['importance'] = imp_path
    
    return vis_paths


# 路由：首页（文件上传）
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 检查是否有文件上传
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # 检查文件是否合法
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # 生成唯一ID，避免文件冲突
            results_id = str(uuid.uuid4())[:8]  # 8位唯一ID
            # 保存上传的文件
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{results_id}_{filename}')
            file.save(upload_path)
            
            # 读取数据
            try:
                df = pd.read_excel(upload_path)
                X = df.values
                feature_names = df.columns.tolist()
            except Exception as e:
                return f"数据读取错误: {str(e)}"
            
            # 运行PCA分析
            try:
                # 1. 确定最优组件数
                pca_full = PCA(n_components=None, scaling='standard')
                pca_full.fit(X, feature_names=feature_names)
                optimal_components = pca_full.get_optimal_components(threshold=0.9)
                
                # 2. 用最优参数运行PCA
                pca = PCA(n_components=optimal_components, scaling='standard')
                X_transformed = pca.fit_transform(X, feature_names=feature_names)
                
                # 3. 生成Excel结果
                excel_path = os.path.join(app.config['RESULTS_FOLDER'], f'{results_id}_pca_results.xlsx')
                pca.export_results_to_excel(X, output_path=excel_path)
                
                # 4. 生成可视化图表
                vis_paths = generate_visualizations(pca, X, X_transformed, results_id)
                
                # 5. 传递结果到展示页面
                return render_template('results.html',
                                      results_id=results_id,
                                      vis_paths=vis_paths,
                                      excel_filename=f'{results_id}_pca_results.xlsx',
                                      n_components=optimal_components,
                                      n_outliers=np.sum(pca.find_outliers(X)),
                                      total_samples=X.shape[0],
                                      reconstruction_error=pca.reconstruction_error(X),
                                      top_features=[pca.feature_names_[i] for i in np.argsort(pca.get_feature_importance())[::-1][:3]]
                                     )
            except Exception as e:
                return f"PCA分析错误: {str(e)}"
    
    # GET请求：显示上传页面
    return render_template('index.html')


# 路由：下载结果文件
@app.route('/results/<filename>')
def download_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename, as_attachment=True)


# 启动应用
if __name__ == '__main__':

    app.run(host='0.0.0.0', port=80) # 开发模式（生产环境需改为app.run(host='0.0.0.0', port=80)）
