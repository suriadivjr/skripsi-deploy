#import depedencies and libraries
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

st.markdown("<h1 style='text-align: center; color: black; font-size: 25px'><i>AUTOMATIC EXTRACTIVE SUMMARIZATION</i> UNTUK BERITA COVID-19 BERBAHASA INDONESIA MENGGUNAKAN <i>WORD2VEC</i> DENGAN ALGORITMA <i>MAXIMUM MARGINAL RELEVANCE</i> (MMR)</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: black; font-size: 18px'>Suriadi Vajrakaruna - 140810180038 - Teknik Informatika FMIPA Universitas Padjadjaran</h6>", unsafe_allow_html=True)

st.text("")
st.text("")

st.markdown("<p style='text-align: left; color: black; font-size: 17px'><i>Corona Virus Disease of 2019</i> (Covid-19) merupakan penyakit yang disebabkan oleh virus SARS-CoV-2 dan dinyatakan menjadi pandemi global oleh <i>World Health Organization</i> (WHO) pada 9 Maret 2020. Hingga Januari 2022, kasus terkonfirmasi positif mencapai 4,29 juta dan menyebabkan sekitar 144.000 kematian. Pandemi yang panjang inilah mengakibatkan hal-hal terkait pandemi ini menjadi kebutuhan informasi masyarakat sehingga berita daring mengenai pandemi Covid-19 tersebar luas. Selain itu, tingkat kewaspadaan masyarakat juga mendorong media untuk mempublikasi berita terkait virus tersebut dalam jumlah besar. Jumlah yang besar dan penyebaran cepat yang terbantu oleh internet memakan waktu dan menyulitkan masyarakat dalam memilah serta membaca keseluruhan berita. Oleh karena itu, penelitian ini diharapkan dapat membantu masyarakat untuk menghemat waktu dengan cara meringkas berita tersebut.</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: left; color: black; font-size: 17px'>Program peringkas otomatis pada web ini dapat menghasilkan ringkasan ekstraktif khusus untuk berita Covid-19. Ringkasan ekstraktif adalah ringkasan yang dihasilkan dari beberapa kalimat teks sumber yang dianggap relevan untuk mewakili seluruh isi dokumen. Jumlah kalimat yang terkandung pada ringkasan sebesar setengah dari jumlah kalimat yang ada pada teks sumber. Program ini dirancang menggunakan bahasa pemrograman <i>Python</i> dengan menggunakan algoritma <i>Maximum Marginal Relevance</i> (MMR) dengan bantuan <i>Word2Vec</i> untuk mengonversi kata-kata yang ada pada teks sumber menjadi vektor. Vektor-vektor tersebut akan dihitung dengan rumus <i>Cosine Similarity</i> untuk menentukan tingkat kerelevanan sebuah kalimat dengan isi keseluruhan dokumen dan perhitungan MMR untuk mendapatkan nilai MMR. Kalimat dengan nilai MMR tertinggi akan terpilih menjadi kalimat yang akan dipilih menjadi kalimat ringkasan.</p>", unsafe_allow_html=True)

st.text("")
st.text("")

st.markdown("<h6 style='text-align: left; color: black; font-size: 22px'>Program Peringkas Otomatis</h6>", unsafe_allow_html=True)
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
            csvfile = open('./programdata/ori' + news_id + '.csv', 'a+', encoding="utf-8")
            csvfile.write("sentence\n")
            ori_sentence_clean = text.replace('\\t', " ").replace('\\n', " ").replace('\\u'," ")
            ori_sentence_split = sent_tokenize(ori_sentence_clean)
            for sentence_split in ori_sentence_split:
                csvfile.write(sentence_split + "\n")
            csvfile.close()


            #split original sentences and write to new csv file "ori.csv"
            csvfile = open('./programdata/ori' + news_id + '.csv', 'a+', encoding="utf-8")
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

            csvfile = open('./programdata/processed' + news_id + '.csv', 'a+', encoding="utf-8")
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

            NEWS_DATA = pd.read_csv('./programdata/processed' + news_id + '.csv')
            nltk.download('punkt')
            nltk.download('stopwords')

            sentence_token = (NEWS_DATA['sentence_cleaned'].apply(str)).apply(word_tokenize_wrapper)
            stopword_removed = sentence_token.apply(remove_stopwords)
            news_token = stopword_removed.apply(get_stemmed_term)
            NEWS_DATA['sentence_processed'] = news_token.apply(join_text_list)
            NEWS_DATA.to_csv('./programdata/processed' + news_id + '.csv')


            #load Word2Vec Model
            model = Word2Vec.load("./w2vmodel.w2v")


            #convert query to vector
            df = pd.read_csv('./programdata/processed' + news_id + '.csv', sep=',').dropna()
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
            n = math.floor(len(sentences) / 2) 
            alpha = 0.7
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
            summarySetSorted = sorted(summarySet)
            news = pd.read_csv('./programdata/ori' + news_id + '.csv', sep=';').dropna()
            summaryResult = []
            for summary in summarySetSorted:
                summaryResult.append(str(news['sentence'][summary].lstrip(' ')))
            summaryText = ' '.join(summaryResult).replace('\\"','"')
            st.text_area("Ringkasan Berita", disabled=True, height=250, max_chars=None, value=summaryText, placeholder="Ringkasan berita Anda akan ditampilkan di sini.")

        except ValueError:
            st.error("Berita Anda tidak mengandung kata-kata yang berarti.")

        except ZeroDivisionError:
            st.error("Anda belum memasukkan berita atau berita yang Anda masukkan kosong.")

        os.remove('./programdata/ori' + news_id + '.csv')
        os.remove('./programdata/processed' + news_id + '.csv')
else:
    st.text_area("Ringkasan Berita", disabled=True, height=250, max_chars=None, value="", placeholder="Ringkasan berita Anda akan ditampilkan di sini.")