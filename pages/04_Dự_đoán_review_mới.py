import streamlit as st
import pandas as pd
import numpy as np
import pickle
from underthesea import word_tokenize, pos_tag, sent_tokenize
import regex


st.set_page_config(page_title="Mô tả dự án",
                   layout="wide",
                   initial_sidebar_state="auto")


st.markdown("# <center>Project 1:<span style='color:#4472C4; font-family:Calibri (Body);font-style: italic;'> Sentiment Analysis</span></center>", unsafe_allow_html=True)

st.subheader("Nhập vào comment", divider='rainbow')

# Cho người dùng chọn nhập dữ liệu hoặc upload file
type = st.radio("Chọn cách nhập dữ liệu", options=["Nhập dữ liệu trực tiếp", "Upload file"])
lst = []
if type == "Nhập dữ liệu trực tiếp":
    comment1 = st.text_input("Text 1")
    if comment1:
        lst.append(comment1)
    comment2 = st.text_input("Text 2")
    if comment2:
        lst.append(comment2)
elif type == "Upload file":
    st.subheader("Upload file")
    # Upload file
    uploaded_file = st.file_uploader("Chọn file dữ liệu", type=["xlsx"])
    if uploaded_file is not None:
        # Đọc file dữ liệu
        df = pd.read_excel(uploaded_file)
        print(df.columns.tolist())
        if len(df.columns) == 1 and df.columns.tolist()[0] == "New_Comment":
            lst = df["New_Comment"].tolist()
        else:
            st.warning('file upload chứa dữ liệu không hợp lệ!', icon="⚠️")
        
        # st.write(df)
    with st.expander("**Yêu cầu file upload như sau:**"):
        st.markdown("""
* File có định dạng **.xlsx**
* Có 1 cột: **New_Comment**
* Ví dụ:

| New_Comment |
|---|
| Món nướng ngon quá!  |
| Đồ ăn hôm nay dở thiệt!  |
                """)
        st.markdown('<div style="padding: 10px 5px;"></div>', unsafe_allow_html=True)
        st.markdown("* Hoặc download **file mẫu** như sau:")
        df_sample = pd.DataFrame({"New_Comment": ["Món nướng ngon quá!",
                                                  "Đồ ăn hôm nay dở thiệt!",
                                                  "Sợ ốc cát nhưng k hề nhé. Rất ok so với giá luôn. Chất lượng ok. Đồ ăn tươi, tôm sống. Có ghẹ nữa.",
                                                  "Giá rẻ. Đồ ăn ngon, ướp thấm vị. Quán sạch sẽ, không gian  thoáng. Nhân viên rất dễ thương, chu đáo.",
                                                  "Quán rộng rãi, đồ ăn được, giá hợp lý, lên món nhiều ăn thoải mái. Không bị phụ thu, tiền nước không quá cao",
                                                  "Bảo vệ quán này không dắt xe khi trời mưa.Trích nguyên văn lời anh bv: “bảo vệ chỉ có ngồi coi xe chứ không có dắt xe”"]})
        # st.dataframe(df_sample, hide_index=True)
        df_sample.to_excel('sample_comment.xlsx', index=False)
        # Tạo download button
        with open('sample_comment.xlsx', 'rb') as f:
            file_contents = f.read()
        st.download_button(
            label="Download sample excel file",
            data=file_contents,
            file_name='sample_comment.xlsx',
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# Chuyển comment vào dataframe
new_df = pd.DataFrame({"New_Comment": lst})
st.dataframe(new_df, width=1200, height=200)

## Xử lý tiếng Việt
##LOAD EMOJICON
file = open('files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
#print(teen_dict)
file.close()
#################
#LOAD TEENCODE
file = open('files/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
#print(teen_dict)
file.close()
###############
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open('files/english-vnmese.txt', 'r', encoding="utf8")
englist_lst = file.read().split('\n')
for line in englist_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
#print(teen_dict)
file.close()
################
#LOAD wrong words. Bỏ 1 số từ liên quan đến "ngon".
file = open('files/wrong-word2.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()
#################
#LOAD STOPWORDS
file = open('files/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()

# Tạo hàm xử lý emoji, teencode, translate, wrong words, stopwords
def process_text(text, dict_emoji, dict_teen, lst_wrong):
    document = text.lower()
    document = document.replace("’",'')
    document = regex.sub(r'\.+', ".", document)
    new_sentence =''
    for sentence in sent_tokenize(document):
        # if not(sentence.isascii()):
        ###### CONVERT EMOJICON
        sentence = ''.join(dict_emoji[word]+' ' if word in dict_emoji else word for word in list(sentence))
        ###### CONVERT TEENCODE
        sentence = ' '.join(dict_teen[word] if word in dict_teen else word for word in sentence.split())
        ###### DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern,sentence))
        ###### DEL wrong words
        sentence = ' '.join('' if word in lst_wrong else word for word in sentence.split())
        new_sentence = new_sentence+ sentence + '. '
    document = new_sentence
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"
    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def covert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

import re
# Hàm để chuẩn hóa các từ có ký tự lặp
def normalize_repeated_characters(text):
    # Thay thế mọi ký tự lặp liên tiếp bằng một ký tự đó
    # Ví dụ: "ngonnnn" thành "ngon", "thiệtttt" thành "thiệt"
    return re.sub(r'(.)\1+', r'\1', text)

def remove_stopword(text, stopwords):
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

## Tiền xử lý Tiếng Việt cho comment mới
# Xử lý emoji, teencode, wrong word
if new_df.size != 0:
    new_df = new_df.assign(Comment_2=lambda x: x['New_Comment'].apply(lambda x: process_text(str(x), emoji_dict, teen_dict, wrong_lst)))
    # Chuẩn hóa unicode
    new_df = new_df.assign(Comment_2=lambda x: x['Comment_2'].apply(lambda x:  covert_unicode(str(x))))
    # Chuẩn hóa ký tự lặp
    new_df = new_df.assign(Comment_2=lambda x: x['Comment_2'].apply(lambda x:  normalize_repeated_characters(str(x))))
    # Loại bỏ stopword
    new_df = new_df.assign(Comment_2=lambda x: x['Comment_2'].apply(lambda x:  remove_stopword(str(x), stopwords_lst)))

def predict_comment(df):
    ## Load model TfIDF đã save
    # Đường dẫn file model Tfidf
    path_tfidf = "data/tfidf02_model.pkl"
    # Load model
    with open(path_tfidf, 'rb') as f:
        tfidf_model = pickle.load(f)
    ## Transform data comment mới
    # Tạo dataframe mới chứa các tfidf features
    df_tfidf02 = pd.DataFrame(tfidf_model.transform(df['Comment_2']).toarray(), columns=tfidf_model.get_feature_names_out())
        
    ## Load model Logistic Regression
    # Đường dẫn file model Logistic Regression  
    path_lr = "data/lr_smote_model.pkl"
    # Load model
    with open(path_lr, 'rb') as f:
        lr_model = pickle.load(f)
        
    ## Dự đoán comment mới
    # Dự đoán
    pred = lr_model.predict(df_tfidf02)
    # Chuyển kết quả vào dataframe
    new_df["pred"] = pred

    new_df["Đánh giá"] = new_df["pred"].map({0: "😡", 1: "👍"})
    ## Load dataframe sau khi xử lý
    st.subheader("Kết quả dự đoán", divider='rainbow')
    st.dataframe(new_df[["New_Comment", "Đánh giá"]],
                column_config={
                    "New_Comment": "Bạn đã nhập",
                    "Đánh giá": {'alignment': 'center'}
                    },
                hide_index=True, width=1200, height=200)
    st.write("*Ghi chú*")
    st.write('👍 = Tích cực, 😡 = Tiêu cực')
    
if new_df.size != 0:
    predict_comment(new_df)

