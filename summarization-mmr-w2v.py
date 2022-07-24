#import depedencies and libraries
from email.mime import image
import streamlit as st
import random as rand
import re
import string
import pandas as pd
import os
import operator
import math
import numpy as np
from numpy import place
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from gensim.models.word2vec import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

st.set_page_config(page_title="140810180038 - Suriadi Vajrakaruna - Skripsi")
st.sidebar.write('Program Peringkas Otomatis Berita Covid-19 Berbahasa Indonesia')
menu_name = st.sidebar.selectbox(
    'Pilih Menu', ['Deskripsi Program', 'Program Peringkas', 'Biodata Penulis']
)

if menu_name == 'Deskripsi Program':
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        img = Image.open('unpad.png')
        st.image(img, use_column_width = 'always')
    st.markdown("<h1 style='text-align: center; color: black; font-size: 25px'><i>AUTOMATIC EXTRACTIVE SUMMARIZATION</i> UNTUK BERITA COVID-19 BERBAHASA INDONESIA MENGGUNAKAN <i>WORD2VEC</i> DENGAN ALGORITMA <i>MAXIMUM MARGINAL RELEVANCE</i> (MMR)</h1>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: black; font-size: 18px'>Suriadi Vajrakaruna - 140810180038 - Teknik Informatika FMIPA Universitas Padjadjaran</h6>", unsafe_allow_html=True)
    st.text("")
    st.text("")
    st.markdown("<p style='text-align: justify; color: black; font-size: 17px'><i>Corona Virus Disease of 2019</i> (Covid-19) merupakan penyakit yang disebabkan oleh virus SARS-CoV-2 dan dinyatakan menjadi pandemi global oleh <i>World Health Organization</i> (WHO) pada 9 Maret 2020. Hingga Januari 2022, kasus terkonfirmasi positif mencapai 4,29 juta dan menyebabkan sekitar 144.000 kematian. Pandemi yang panjang inilah mengakibatkan hal-hal terkait pandemi ini menjadi kebutuhan informasi masyarakat sehingga berita daring mengenai pandemi Covid-19 tersebar luas. Selain itu, tingkat kewaspadaan masyarakat juga mendorong media untuk mempublikasi berita terkait virus tersebut dalam jumlah besar. Jumlah yang besar dan penyebaran cepat yang terbantu oleh internet memakan waktu dan menyulitkan masyarakat dalam memilah serta membaca keseluruhan berita. Oleh karena itu, penelitian ini diharapkan dapat membantu masyarakat untuk menghemat waktu dengan cara meringkas berita tersebut.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: justify; color: black; font-size: 17px'>Program peringkas otomatis pada web ini dapat menghasilkan ringkasan ekstraktif khusus untuk berita Covid-19. Ringkasan ekstraktif adalah ringkasan yang dihasilkan dari beberapa kalimat teks sumber yang dianggap relevan untuk mewakili seluruh isi dokumen. Jumlah kalimat yang terkandung pada ringkasan sebesar setengah dari jumlah kalimat yang ada pada teks sumber. Program ini dirancang menggunakan bahasa pemrograman <i>Python</i> dengan menggunakan algoritma <i>Maximum Marginal Relevance</i> (MMR) dengan bantuan <i>Word2Vec</i> untuk mengonversi kata-kata yang ada pada teks sumber menjadi vektor. Vektor-vektor tersebut akan dihitung dengan rumus <i>Cosine Similarity</i> untuk menentukan tingkat kerelevanan sebuah kalimat dengan isi keseluruhan dokumen dan perhitungan MMR untuk mendapatkan nilai MMR. Kalimat dengan nilai MMR tertinggi akan terpilih menjadi kalimat yang akan dipilih menjadi kalimat ringkasan.</p>", unsafe_allow_html=True)

    st.text("")
    st.text("")

if menu_name == 'Program Peringkas':
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        img = Image.open('unpad.png')
        st.image(img, use_column_width = 'always')
    st.markdown("<h1 style='text-align: center; color: black; font-size: 25px'>Program Peringkas Otomatis Berita Covid-19</h1>", unsafe_allow_html=True)
    stralpha = st.selectbox("Pilih Lambda (λ). Nilai optimal berada pada 0,7 atau 0,8. Semakin kecil nilai λ, kalimat yang diekstrak akan lebih bervariasi. Semakin besar λ, kalimat yang diekstrak akan lebih mengacu ke ide pokok berita.", 
                        ('0.7', '0.8', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.9'))
    strn = st.number_input("Jumlah kalimat ringkasan yang diinginkan:", min_value=1)
    text_inp = st.text_area("Berita Orisinal", disabled=False, height=250, max_chars=None, placeholder="Masukkan berita yang ingin Anda ringkas di sini.", key="input")
    

    col1, col2, col3 , col4, col5 = st.columns(5)
    with col1:
        pass
    with col2:
        pass
    with col4:
        pass
    with col5:
        pass
    with col3 :
        center_button = st.button('Ringkas Berita')

    if center_button:
        text = str(text_inp.replace('"','\\"'))
        dots = text.count('.')

        if dots < 2:
            st.error("Anda belum memasukkan berita atau berita yang Anda masukkan terlalu pendek.")
        else:
            try:
                #generate random ID 
                news_id = str(rand.randint(0, 10000000))

                #split original sentences and write to new csv file "ori.csv"
                csvfile = open('./ori' + news_id + '.csv', 'a+', encoding="utf-8")
                csvfile.write("sentence\n")
                ori_sentence_clean = text.replace('\\t', " ").replace('\\n', " ").replace('\\u'," ")
                ori_sentence_split = sent_tokenize(ori_sentence_clean)
                for sentence_split in ori_sentence_split:
                    csvfile.write(sentence_split + "\n")
                csvfile.close()


                #split original sentences and write to new csv file "ori.csv"
                csvfile = open('./ori' + news_id + '.csv', 'a+', encoding="utf-8")
                csvfile.write("sentence\n")
                ori_sentence_clean = text.replace('\\t', " ").replace('\\n', " ").replace('\\u'," ")
                ori_sentence_split = sent_tokenize(ori_sentence_clean)
                for sentence_split in ori_sentence_split:
                    csvfile.write(sentence_split + "\n")
                csvfile.close()


                #clean sentence and write cleaned sentences to new csv file "processed.csv"
                def remove_special(text):
                    text = text.replace('\\t', " ").replace('\\n', " ").replace('\\u'," ").replace('\\', "")
                    text = text.encode('ascii', 'replace').decode('ascii')
                    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text).split())
                    text = text.replace("http://", " ").replace("https://", " ")
                    return text.replace("?", " ")

                def remove_punctuation(text):
                    punctuation = '''!()[]{};:.'"\,<>?@#$%^&*_~'''
                    punctuation_space = '''-/'''
                    for char in text:
                        if char in punctuation:
                            text = text.replace(char, "")
                        elif char in punctuation_space:
                            text = text.replace(char, " ")
                    return text

                def remove_number(text):
                    return ' '.join(re.sub("([+-]?([0-9]*[.])?[0-9]+)|([+-]?([0-9]*[,])?[0-9]+)", "", text).split())

                def remove_single_char(text):
                    return re.sub(r"\b[a-zA-Z]\b", "", text)

                def text_cleaning(text):
                    rem_punc = remove_punctuation(text)
                    case_folded = rem_punc.lower()
                    special_removed = remove_special(case_folded)
                    number_removed = remove_number(special_removed)
                    single_char_removed = remove_single_char(number_removed)
                    return single_char_removed

                csvfile = open('./processed' + news_id + '.csv', 'a+', encoding="utf-8")
                csvfile.write("sentence_cleaned,sentence_processed\n")
                for sentence_split in ori_sentence_split:
                    cleaned_text = text_cleaning(sentence_split)
                    csvfile.write(cleaned_text+"\n")
                csvfile.close()


                #process sentence and write processed sentences to existing csv file "processed.csv"
                def word_tokenize_wrapper(text):
                    return word_tokenize(text)

                sastrawi_stopword_factory = StopWordRemoverFactory()
                LIST_STOPWORDS = stopwords.words('indonesian') + sastrawi_stopword_factory.get_stop_words()
                LIST_STOPWORDS.extend(['gettyimages', 'getty', 'imagesdengan', 'loh', '&amp', 'yah', 'ga', 'gak', 'engga', 'ngga', 'aja', 'nih', 'sih', 'pada'])
                def remove_stopwords(words):
                    return [word for word in words if word not in LIST_STOPWORDS]

                factory = StemmerFactory()
                stemmer = factory.create_stemmer()

                def get_stemmed_term(words):
                    return [stemmer.stem(word) for word in words]

                def join_text_list(words):
                    return ' '.join([word for word in words])

                NEWS_DATA = pd.read_csv('./processed' + news_id + '.csv')
                nltk.download('punkt')
                nltk.download('stopwords')

                sentence_token = (NEWS_DATA['sentence_cleaned'].apply(str)).apply(word_tokenize_wrapper)
                stopword_removed = sentence_token.apply(remove_stopwords)
                news_token = stopword_removed.apply(get_stemmed_term)
                NEWS_DATA['sentence_processed'] = news_token.apply(join_text_list)
                NEWS_DATA.to_csv('./processed' + news_id + '.csv')


                #load Word2Vec Model
                model = Word2Vec.load("./w2vmodel.w2v")


                #convert query to vector
                df = pd.read_csv('./processed' + news_id + '.csv', sep=',').dropna()
                df.rename(columns = {'Unnamed: 0':'id'}, inplace = True)

                oneSentenceDoc = ''
                oneSentenceDoc = df["sentence_processed"].str.cat(sep=' ') #using string
                tokens = oneSentenceDoc.split()

                wordvectors = model.wv
                vectors = []
                for word in tokens:
                    if word in model.wv:
                        vectors.append(wordvectors.get_vector(word).reshape(1,-1))
                    else:
                        vectors.append(np.zeros(model.vector_size))
                doc_vector = sum(vectors) / len(vectors)

                #convert every sentence to vector
                sentences = df["sentence_processed"]

                sentences_vector = {}
                for sentence in sentences:
                    tokens = sentence.split()
                    vectors = []
                    for word in tokens:
                        if word in model.wv:
                            vectors.append(wordvectors.get_vector(word).reshape(1,-1))
                        else:
                            vectors.append(np.zeros(model.vector_size))
                    avg_value = sum(vectors) / len(vectors)
                    sentence_id = df.index[df["sentence_processed"]==sentence]
                    sentences_vector[sentence_id[0]] = avg_value


                #count cosine similarity between query and each sentence
                #prepare cosine similarity function for MMR equation
                cs_scores = {}
                for i in range(0,len(sentences_vector)):
                    cs_scores[i] = cosine_similarity(doc_vector, sentences_vector[i])

                def calculate_similarity(sentence1, sentence2):
                    wordvectors = model.wv
                    if sentence2 == []:
                        return 0
                    
                    vectors = []
                    tokens = str(sentences[sentence1]).split()
                    for word in tokens:
                        if word in model.wv:
                            vectors.append(wordvectors.get_vector(word).reshape(1,-1))
                        else:
                            vectors.append(np.zeros(model.vector_size))
                    sentence1_vector = sum(vectors) / len(vectors)
                    
                    tokens.clear()
                    vectors.clear()
                    tokens = str(sentences[sentence2[0]]).split()
                    for word in tokens:
                        if word in model.wv:
                            vectors.append(wordvectors.get_vector(word).reshape(1,-1))
                        else:
                            vectors.append(np.zeros(model.vector_size))
                    sentence2_vector = sum(vectors) / len(vectors)
                    
                    return cosine_similarity(sentence1_vector, sentence2_vector)[0][0]


                #process MMR score
                n = int(strn)
                if n > len(sentences):
                    st.error('Jumlah kalimat ringkasan tidak boleh melebihi jumlah kalimat berita orisinal.')        
                    summarySet = None
                else:
                    alpha = float(stralpha)
                    summarySet = []
                    while n > 0:
                        mmr = {}
                        for sentence in cs_scores.keys():
                            if not sentence in summarySet:
                                mmr[sentence] = alpha * cs_scores[sentence] - (1-alpha) * calculate_similarity(sentence, summarySet)
                        selected = max(mmr.items(), key=operator.itemgetter(1))[0]
                        summarySet.append(selected)
                        n -= 1


                #extract summary from existing csv file "ori.csv" based on ID in summarySetSorted
                if summarySet is not None:
                    summarySetSorted = sorted(summarySet)
                    news = pd.read_csv('./ori' + news_id + '.csv', sep=';').dropna()
                    summaryResult = []
                    for summary in summarySetSorted:
                        summaryResult.append(str(news['sentence'][summary].lstrip(' ')))
                    summaryText = ' '.join(summaryResult).replace('\\"','"')
                    st.text_area("Ringkasan Berita", disabled=True, height=250, max_chars=None, value=summaryText, placeholder="Ringkasan berita Anda akan ditampilkan di sini.")

            except ValueError:
                st.error("Berita Anda tidak mengandung kata-kata yang berarti.")

            except ZeroDivisionError:
                st.error("Berita yang Anda masukkan kosong.")

            os.remove('./ori' + news_id + '.csv')
            os.remove('./processed' + news_id + '.csv')
    else:
        st.text_area("Ringkasan Berita", disabled=True, height=250, max_chars=None, value="", placeholder="Ringkasan berita Anda akan ditampilkan di sini.")

if menu_name == 'Biodata Penulis':
    st.markdown("<h1 style='text-align: center; color: black; font-size: 25px'>Biodata Penulis</h1>", unsafe_allow_html=True)
    st.write(' ')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        image = Image.open('fotoku.jpg')
        st.image(image, use_column_width = 'always', caption='Suriadi Vajrakaruna')
    with col3:
        st.write(' ')
    st.markdown("<p style='text-align: justify; color: black; font-size: 17px'><b>Suriadi Vajrakaruna</b> merupakan penulis skripsi ini. Lahir pada 25 Juni 2000 di Kota Tangerang, penulis merupakan anak pertama dari tiga bersaudara. Sejak lahir, penulis berdomisili di Kota Tangerang dan menempuh SD, SMP, dan SMA berturut-turut pada SD Anak Terang, SMP Pahoa, dan SMA Negeri 2 Kota Tangerang. Saat ini, penulis sedang menempuh pendidikan S1 pada Teknik Informatika FMIPA Universitas Padjadjaran. Namun, penulis telah memiliki beberapa pengalaman yang diperoleh dari program magang sebagai <i>Quality Assurance Tester & Engineer</i> pada Campaign.com dan <i>Product Manager</i> pada AwanTunai. Selama masa kuliah, penulis mendapat beberapa prestasi apresiasi seperti <i>Wallstreet English Level 15 Completion</i> dan Juara 2 KTI TIK pada ajang Gemastik XII pada 2019 serta Juara I KTK TIK pada ajang Gemastik XIII, Mahasiswa Berprestasi Teknik Informatika Unpad, dan dinominasikan sebagai Mahasiswa dengan Publikasi Riset Terproduktif Unpad Awards pada tahun 2020.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: justify; color: black; font-size: 17px'>Selain apresiasi dan prestasi, penulis juga turut aktif dalam beberapa organisasi dan kepanitiaan. Mulai dari tahun 2018, pada saat penulis menjadi mahasiswa baru, penulis telah dipercaya untuk menjadi Ketua Divisi Humas untuk Musyawarah Besar IV Teknik Informatika Unpad. Setelah itu, penulis menjabat sebagai Anggota Komisi I DPA Himatif FMIPA Unpad serta merangkap sebagai Presidium I Himatif FMIPA Unpad selama setahun sampai 2019. Di luar himpunan mahasiswa, penulis juga mengambil andil sebagai anggota Divisi <i>Roadshow</i> untuk acara PARADE 2019 yang diselenggarakan oleh Paguyuban Mahasiswa Banten Unpad (PAMATEN Unpad). Pada pertengahan tahun 2019 sampai tahun 2020, penulis menjabat sebagai ketua Divisi Band untuk KKM Musik Artemipa FMIPA Unpad. Selanjutnya, penulis bertanggung jawab atas lomba musikalisasi puisi untuk ajang OSEAN 2020 FMIPA Unpad. Setelah OSEAN selesai, penulis aktif dalam Keluarga Mahasiswa Buddhist Dharmavira Unpad (KMBD Unpad) sebagai staff Divisi Hubungan Eksternal selama setengah tahun. Terakhir, pada tahun 2021 sampai 2022, penulis menjadi Asisten Praktikum untuk Departemen Ilmu Komputer Unpad.</p>", unsafe_allow_html=True)