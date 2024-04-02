import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

st.set_page_config(page_title="Mô tả dự án",
                   layout="wide",
                   initial_sidebar_state="auto")

st.markdown("# <center>Project 1:<span style='color:#4472C4; font-family:Calibri (Body);font-style: italic;'> Sentiment Analysis</span></center>", unsafe_allow_html=True)


col1, padding1, col2, padding2, col3 = st.columns((8,1,20,1,8))
with col1:
    col1.write(' '*10)
    col1.write(' '*10)
    col1.write(' '*10)
    col1.write(' '*10)
    col1.image("images/like.png")
with col2:
    col2.image("images/shopeefood.jpg")
with col3:
    col3.write(' '*10)
    col3.write(' '*10)
    col3.write(' '*10)
    col3.write(' '*10)
    col3.image("images/dislike.png")

st.write('<center><span style="font-weight:bolder;color:#4472C4">Giáo viên hướng dẫn: ThS. Khuất Thùy Phương</span><center>', unsafe_allow_html=True)
st.write('<center><span style="font-weight:bolder;color:#4472C4">Học viên: Trần Hạnh Triết</span><center>', unsafe_allow_html=True)
st.write('<center><span style="font-weight:bolder;color:#4472C4">Bạn cùng nhóm: Nguyễn Minh Thức</span><center>', unsafe_allow_html=True)

st.subheader("Mục tiêu", divider='rainbow')
st.markdown("##### Xây dựng mô hình dự đoán bình luận của của khách hàng về nhà hàng/ quán ăn (tích cực, tiêu cực hay trung tính), điều này giúp cho nhà hàng hiểu được tình hình kinh doanh, giúp nhà hàng cải thiện hơn trong dịch vụ, sản phẩm"
)

st.subheader("Dữ liệu", divider='rainbow')
type = st.selectbox("Chọn dataframe", ("1_Restaurants.csv", "2_Reviews.csv"))
if type == "1_Restaurants.csv":
    res = pd.read_csv('data/1_Restaurants.csv', encoding='utf8', sep=',')
    st.dataframe(res, hide_index=True, height=200)
elif type == "2_Reviews.csv":
    review_data = pd.read_csv('data/2_Reviews.csv', encoding='utf8', sep=',')
    st.dataframe(review_data, hide_index=True, height=200)

st.subheader("EDA", divider='rainbow')
col1, padding, col2 = st.columns((30,1,45))
with col1:
    col1.write('<center><span style="font-weight:bold;">Số đơn hàng mỗi quận</span><center>', unsafe_allow_html=True)
    col1.image("images/dist_invoice.png")
with col2:
    col2.write('<center><span style="font-weight:bold;">Phân phối Ratings</span></center>', unsafe_allow_html=True)
    col2.image("images/hist_rating.png")

st.subheader("Tiền xử lý tiếng Việt", divider='rainbow')
col1, col2, col3 = st.columns((5,50,5))
with col1:
    col1.write(' '*10)
with col2:
    col2.image('images/tienxulytiengviet.png')
with col3:
    col3.write(' '*10)

st.subheader("Set tập luật", divider='rainbow')
st.write("* Tích cực: Positive_count >= Negative_count")
st.write("* Tiêu cực: Positive_count < Negative_count")
type = st.selectbox("Danh sách từ", ("positive_words", "negative_words"))
if type == "positive_words":
    pos = '''positive_words = [
    "thích", "tốt", "xuất sắc", "tuyệt vời", "tuyệt hảo", "đẹp", "ổn", "ngon",
    "hài lòng", "ưng ý", "hoàn hảo", "chất_lượng", "thú vị", "nhanh", "nghiện",
    "cảm động", "phục vụ tốt", "làm hài lòng", "gây ấn tượng", "nổi trội",
    "sáng tạo", "quý báu", "phù hợp", "tận tâm",
    "hiếm có", "cải thiện", "hoà nhã", "chăm chỉ", "cẩn thận",
    "vui vẻ", "sáng sủa", "hào hứng", "đam mê", "vừa vặn", "đáng tiền",
    "ok", "không tệ", "ổn", "mlem", "nhiệt tình", "chu đáo", "niềm nở", "mềm", "ẩm",
    "tươi", "đậm_đà", "sạch sẽ", "hợp lý", "rẻ", "bình dân", "vừa phải", "sáng",
    "thoáng", "rộng", "mát", "giòn", "vừa miệng", "vừa vị", "hết xẩy", "đậm vị",
    "mlem", "không_bị", "bắt_mắt", "giòn", "nghiêm_túc", "đầy_ắp"]'''
    st.code(pos, language='python')
elif type == "negative_words":
    neg = '''negative_words = [
    "kém", "tệ", "đau", "xấu", "dở", "ức", "tức_giận", "tức", "đắt", "lừa",
    "buồn", "rối", "thô lỗ", "cộc", "lâu", "chán", "đuổi", "cọc_lóc",
    "không_ấn tượng", "không_ngon", "chẳng_ngon", "hết ngon" "chậm", "khó khăn", "phức tạp",
    "không_tiện lợi", "không_đáng tiền", "dởm", "mắc", "chém", "lâu",
    "quạu", "cáu", "bực", "lạt", "nhạt", "nguội", "tanh", "mặn", "cứng",
    "ngán", "ớn", "hôi", "hầm", "thiu", "dai", "bở", "tanh", "cứng", "thiếu",
    "thất vọng", "lừa_đảo", "dơ", "mất vệ sinh", "ghê",
    "đanh đá", "mặt_mày", "không_thích_vị", "không_thích_mùi",
    "bức_xúc", "chê", "không_chất_lượng", "cao", "không_xuất_sắc",
    "không_ưa", "không_thèm", "kì_kì", "ghê_gớm", "ngộ_độc", "tào_tháo",
    "không_tươi", "chút_xíu", "giả_dối", "bức_xúc", "không_giống"]'''
    st.code(neg, language='python')
    
col1, padding, col2 = st.columns((40,1,40))
with col1:
    col1.write('<center><span style="font-weight:bolder;color:blue">Trước xử lý</span><center>', unsafe_allow_html=True)
    bef = pd.read_csv('data/2_Reviews_clean09.csv', encoding='utf8', sep=',')
    bef2 = bef[['IDRestaurant', 'User', 'Rating', 'Comment']]
    col1.dataframe(bef2, hide_index=True, width=500, height=200)
with col2:
    col2.write('<center><span style="font-weight:bolder;color:blue">Sau xử lý</span></center>', unsafe_allow_html=True)
    aft = bef[['IDRestaurant', 'User','Comment_2', 'Positive_Count', 'Negative_Count', 'Sentiment']]
    col2.dataframe(aft, hide_index=True, width=500, height=200)
    # col2.image("images/sauxuly.png")

st.subheader("Build model dự đoán Sentiment", divider='rainbow')
st.write("* TfidfVectorizer")
st.write("* SMOTE: xử lý mất cân bằng dữ liệu")
st.write("* Machine Learning: KNN, Logistic Regression, Decision Tree, Random Forest")
st.write("* PySpark: Logistic Regression, Random Forest")

