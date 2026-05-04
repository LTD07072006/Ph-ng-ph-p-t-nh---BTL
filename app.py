import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, RationalQuadratic, ExpSineSquared
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go

# 1. CẤU HÌNH & KHỞI TẠO BỘ NHỚ TRẠNG THÁI
st.set_page_config(page_title="Hệ thống Dự báo Kriging Pro", layout="wide", page_icon="🚀")

if 'page' not in st.session_state:
    st.session_state.page = 'dashboard' # Trang mặc định là Đồ thị

def chuyen_trang(ten_trang):
    st.session_state.page = ten_trang

# 2. THANH SIDEBAR CHUNG (Xuất hiện ở cả 2 màn hình)
st.sidebar.title("🛠️ Điều khiển Hệ thống")
st.sidebar.markdown(f"**Người dùng:** 24022630 - Lê Tuấn Dũng") # Gợi nhắc ngữ cảnh học tập

# Nút chuyển trang nhanh
if st.session_state.page == 'dashboard':
    st.sidebar.button("💻 Chuyển sang Chế độ Vòng lặp (Log)", on_click=chuyen_trang, args=('console',), use_container_width=True)
else:
    st.sidebar.button("📈 Quay lại Đồ thị Trực quan", on_click=chuyen_trang, args=('dashboard',), use_container_width=True)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📤 Nạp dữ liệu CSV (X, Y)", type="csv")

# =====================================================================
# MÀN HÌNH 1: ĐỒ THỊ PRO + AUTOML
# =====================================================================
if st.session_state.page == 'dashboard':
    st.title("📈 Dự báo Kriging Trực quan (Bản Pro)")
    
    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            X_train = df[['X']].values
            y_train = df['Y'].values

            # Khu vực cài đặt chuyên sâu
            st.sidebar.header("⚙️ Cấu hình AI")
            auto_tune = st.sidebar.toggle("🤖 Bật Tự động Tối ưu (AutoML)", value=False)
            
            kernel_choice = st.sidebar.selectbox(
                "Hàm nhân (Kernel)", 
                ("RBF", "Matern", "Rational Quadratic", "Periodic"), 
                disabled=auto_tune
            )
            ls_val = st.sidebar.slider("Length-scale", 0.1, 10.0, 1.0, 0.1, disabled=auto_tune)
            alpha_val = st.sidebar.slider("Nhiễu (Alpha)", 1e-10, 1.0, 1e-10, 0.1, disabled=auto_tune)

            # Khởi tạo Kernel
            if kernel_choice == "RBF": kernel = ConstantKernel(1.0) * RBF(ls_val)
            elif kernel_choice == "Matern": kernel = ConstantKernel(1.0) * Matern(ls_val, nu=1.5)
            elif kernel_choice == "Rational Quadratic": kernel = ConstantKernel(1.0) * RationalQuadratic(ls_val)
            else: kernel = ConstantKernel(1.0) * ExpSineSquared(ls_val)

            # Huấn luyện
            if auto_tune:
                # n_restarts_optimizer=10 giúp tìm điểm tối ưu tốt hơn
                gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha_val, n_restarts_optimizer=10)
                with st.spinner("🤖 AutoML đang thực hiện hàng trăm phép tính để tối ưu..."):
                    gpr.fit(X_train, y_train)
                
                # --- ĐOẠN CHIẾT XUẤT VÀ IN KẾT QUẢ TỐI ƯU ---
                st.markdown("### 🏆 Kết quả tối ưu từ AutoML")
                
                # Trích xuất thông số từ kernel đã tối ưu
                # Lưu ý: gpr.kernel_ là object sau tối ưu, gpr.kernel là object ban đầu
                params = gpr.kernel_.get_params()
                
                # Lấy Length-scale (nằm trong k2 nếu kernel là Constant * RBF/Matern...)
                optimized_ls = params.get('k2__length_scale', "Không xác định")
                
                # Hiển thị bảng báo cáo nhanh
                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    st.success(f"**Length-scale tối ưu:**\n{optimized_ls:.4f}" if isinstance(optimized_ls, float) else f"**LS:** {optimized_ls}")
                with res_col2:
                    # Alpha (Noise) là giá trị bạn truyền vào cố định cho GPR
                    st.info(f"**Mức nhiễu (Alpha):**\n{alpha_val:.4f}")
                with res_col3:
                    # Tính toán sai số cuối cùng
                    y_opt_pred = gpr.predict(X_train)
                    final_rmse = np.sqrt(mean_squared_error(y_train, y_opt_pred))
                    st.warning(f"**RMSE tối ưu:**\n{final_rmse:.4f}")
                
                st.write(f"📝 **Cấu trúc Kernel cuối cùng:** `{gpr.kernel_}`")
                st.markdown("---")
                
            else:
                # Chế độ thủ công giữ nguyên
                gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha_val, optimizer=None)
                gpr.fit(X_train, y_train)

            # Vẽ đồ thị
            X_plot = np.linspace(df['X'].min(), df['X'].max() + 2, 200).reshape(-1, 1)
            y_pred, sigma = gpr.predict(X_plot, return_std=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.concatenate([X_plot[:,0], X_plot[::-1,0]]), 
                                     y=np.concatenate([y_pred - 1.96*sigma, (y_pred + 1.96*sigma)[::-1]]),
                                     fill='toself', fillcolor='rgba(0,100,255,0.15)', line=dict(color='rgba(0,0,0,0)'), name='95% Confidence'))
            fig.add_trace(go.Scatter(x=X_plot[:,0], y=y_pred, mode='lines', name='Dự báo', line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=X_train[:,0], y=y_train, mode='markers', name='Thực tế', marker=dict(color='red', size=8)))
            
            st.plotly_chart(fig, use_container_width=True)

            # Thông số sai số
            c1, c2 = st.columns(2)
            y_t_p = gpr.predict(X_train)
            c1.metric("RMSE", f"{np.sqrt(mean_squared_error(y_train, y_t_p)):.4f}")
            c2.metric("MAE", f"{mean_absolute_error(y_train, y_t_p):.4f}")

        except Exception as e:
            st.error(f"Lỗi: {e}")
    else:
        st.info("💡 Mời bạn tải file CSV ở thanh bên trái để bắt đầu.")

# =====================================================================
# MÀN HÌNH 2: TRUY VẾT VÒNG LẶP (LOG)
# =====================================================================
else:
    st.title("💻 Nhật ký Vòng lặp Thuật toán")
    
    if uploaded_file is not None:
        st.warning("Chế độ này sẽ quét qua các giá trị Length-scale để bạn thấy quá trình hội tụ của sai số.")
        if st.button("🚀 Bắt đầu quét nghiệm (Grid Search Iteration)", type="primary"):
            uploaded_file.seek(0)
            df_loop = pd.read_csv(uploaded_file)
            X_l, y_l = df_loop[['X']].values, df_loop['Y'].values
            
            console = st.empty()
            log = "Khởi tạo vòng lặp...\n"
            best_r = float('inf')
            
            # Giả lập vòng lặp Phương pháp tính
            test_steps = np.linspace(0.1, 5.0, 20)
            for i, val in enumerate(test_steps):
                m = GaussianProcessRegressor(kernel=ConstantKernel(1.0)*RBF(val), optimizer=None).fit(X_l, y_l)
                err = np.sqrt(mean_squared_error(y_l, m.predict(X_l)))
                
                status = " <-- [NEW BEST]" if err < best_r else ""
                if err < best_r: best_r = err
                
                log += f"Lần lặp {i+1:02d}: LS={val:.3f} | RMSE={err:.4f}{status}\n"
                console.code(log, language='bash')
                time.sleep(0.15) # Tạo hiệu ứng chờ
            
            st.balloons()
            st.success(f"Hoàn tất! Sai số nhỏ nhất tìm được: {best_r:.4f}")
    else:
        st.error("Bạn cần tải file ở Sidebar trước khi chạy vòng lặp!")