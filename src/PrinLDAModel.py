import datetime
import logging
import re
import time
import warnings
from random import randint
import itertools
import gensim
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import seaborn
import spacy
from pandas import DataFrame
from py2neo import Graph
# Sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import chi2
import numpy
import re
import string

import nltk


warnings.filterwarnings("ignore", category=DeprecationWarning)


class PrinLDAModel:
    logging.basicConfig(filename='PrinLDAMOdel_' + str(datetime.datetime.now()) + '.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S', filemode='w')

    nlp = spacy.load('en', disable=['parser', 'ner'])

    best_lda_model = None

    vectorizer = CountVectorizer(analyzer='word',
                                 min_df=2,  # minimum reqd occurences of a word
                                 stop_words='english',  # remove stop words
                                 lowercase=True,  # convert all words to lowercase
                                 token_pattern='[a-zA-Z0-9]{3,}'  # num chars > 3
                                 # max_features=5,  # max number of uniq words
                                 )

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                            stop_words='english')

    neo_graph = Graph(host="localhost", password="124578")

    def label_tweets_all(self):
        logging.info('Start label_tweets_all: ' + time.strftime("%d %m %Y %H %M %S"))
        df_users = self.neo_graph.run("MATCH (u:User)-[p:POSTS]->(t:Tweet) RETURN u.id").to_data_frame()

        df_users.to_csv("Users.csv")

        for index, row in df_users.iterrows():
            self.label_tweet_user(row["u.id"])

        print('End: ' + time.strftime("%d %m %Y %H %M %S"))

    def label_tweet_user(self, id_user):
        logging.info("---Labeling tweets of " + str(id_user))

        df_data = self.data_extration(id_user)

        if df_data.__len__() > 0:
            df_predicted = self.predict_topic_df(df_data)
        else:
            print("No tweets")

        for index, row in df_predicted.iterrows():
            # Store in the DB
            self.neo_graph.run("MATCH (t:Tweet) WHERE t.id={id} SET t.topic = {topic}",
                               {"id": str(index), "topic": int(row['dominant_topic'])})

    def data_extration(self, id_user):

        df = self.neo_graph.run("MATCH (u:User)-[p:POSTS]->(t:Tweet) WHERE u.id={id} RETURN t.id, t.text",
                                {"id": str(id_user)}).to_data_frame()

        # print df.head(5)

        # print("------Data extracted------")

        return df

    def df_cleaning(self, df):
        # Convert to list
        data = df['t.text'].values.tolist()

        # Remove Emails
        data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

        # Remove URLS
        data = [re.sub(r"http\S+", '', sent) for sent in data]

        # print(data[:1])
        data_words = list(self.sent_to_words(data))

        # Do lemmatization keeping only Noun, Adj, Verb, Adverb
        data_lemmatized, data_empty = self.lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        df_index_empty = DataFrame(df["t.id"].iloc[data_empty])

        data_index = df['t.id'].drop(df.index[data_empty])

        # print("------Data cleaned------")
        return data_index, data_lemmatized, df_index_empty

    def sent_to_words(self, sentences):
        for sentence in sentences:
            yield (gensim.utils.simple_preprocess(str(sentence.encode('utf-8')),
                                                  deacc=True))  # deacc=True removes punctuations

    def lemmatization(self, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []

        texts_empty = []

        i = 0
        for sent in texts:
            if sent:
                doc = self.nlp(" ".join(sent))
                result = " ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if
                                   token.pos_ in allowed_postags])
                if result:
                    texts_out.append(result)
                else:
                    texts_empty.append(i)
            else:
                texts_empty.append(i)

            i += 1

        return texts_out, texts_empty

    def model_selection(self, id_users=None, n_components=None):

        if id_users is None:
            df = self.neo_graph.run("MATCH (t:Tweet) RETURN t.id, t.text").to_data_frame()
        else:
            if len(id_users) > 1:
                id_users_str = "\",\"".join(map(str, id_users))
                id_users_str = "[\"" + id_users_str + "\"]"
            else:
                id_users_str = str(id_users)

            query_tweets = 'WITH ' + id_users_str + ' AS arr ' \
                                                    'MATCH (u:User)- [:POSTS]->(t:Tweet) ' \
                                                    'WHERE u.id in arr ' \
                                                    'RETURN t.id, t.text'

            df = self.neo_graph.run(query_tweets).to_data_frame()

        # Cleaning the data
        data_index, data_lemmatized, data_empty = self.df_cleaning(df)

        data_vectorized = self.vectorizer.fit_transform(data_lemmatized)

        # ---- For visualization of the data in a pandas Dataframe
        # # Materialize the sparse data
        # data_dense = data_vectorized.todense()
        #
        # # Compute Sparsicity = Percentage of Non-Zero cells
        # print("Sparsicity: ", ((data_dense > 0).sum() / data_dense.size) * 100, "%")

        # Define Search Param
        if n_components is None:
            search_params = {'n_components': [2, 3, 4, 5], 'learning_decay': [.5, .7, .9]}

            # Init the Model
            lda_model = LatentDirichletAllocation(n_jobs=-1)

            # Init Grid Search Class
            model = GridSearchCV(lda_model, param_grid=search_params)

            # Do the Grid Search
            model.fit(data_vectorized)

            # Best Model
            self.best_lda_model = model.best_estimator_

            print("------Best model selected------")
            # Model Parameters
            print("Best Model's Params: ", model.best_params_)

            # Log Likelihood Score
            print("Best Log Likelihood Score: ", model.best_score_)

            # Perplexity
            print("Model Perplexity: ", self.best_lda_model.perplexity(data_vectorized))


        else:
            # Build LDA Model
            lda_model = LatentDirichletAllocation(n_components=n_components,  # Number of topics
                                                  learning_method='online',
                                                  random_state=100,  # Random state
                                                  batch_size=128  # n docs in each learning iter
                                                  # evaluate_every=-1,  # compute perplexity every n iters, default: Don't
                                                  # n_jobs=-1,  # Use all available CPUs
                                                  )

            self.best_lda_model = lda_model

            self.best_lda_model.fit(data_vectorized)

        # -------------------------For validation ---------------------

        # Document - Topic Matrix
        lda_output = self.best_lda_model.transform(data_vectorized)

        # column names
        topicnames = ["Topic" + str(i) for i in range(self.best_lda_model.n_components)]

        # Make the pandas dataframe
        df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames)

        df_document_topic['dominant_topic'] = df_document_topic.apply(self.get_topic, axis=1)

        # print (df_document_topic.head(10))

        # Topic - Document Distribution
        df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
        df_topic_distribution.columns = ['Topic Num', 'Num Documents']
        # print ("Topic distribution....")
        # print (df_topic_distribution)

        # Topic - Keyword Matrix
        df_topic_keywords = pd.DataFrame(self.best_lda_model.components_)
        # Assign Column and Index
        df_topic_keywords.columns = self.vectorizer.get_feature_names()
        df_topic_keywords.index = topicnames
        # View
        # print ("Topic - Keyword  distribution....")
        # print(df_topic_keywords.head(10))

        df_topic_keywords = df_topic_keywords.T
        df_topic_keywords.to_csv('../results/'+'Topic - Keyword  distribution.csv', encoding='utf-8')

    def predict_topic(self, text):

        # Step 1: Clean with simple_preprocess
        mytext_2 = gensim.utils.simple_preprocess(str(text.encode('utf-8')), deacc=True)

        # Step 2: Lemmatize
        mytext_3 = []
        allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
        doc = self.nlp(" ".join(mytext_2))
        mytext_3.append(" ".join(
            [token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))

        # Step 3: Vectorize transform
        mytext_4 = self.vectorizer.transform(mytext_3)

        # Step 4: LDA Transform
        topic_probability_scores = self.best_lda_model.transform(mytext_4)
        topic = np.argmax(topic_probability_scores)

        print("predicted topic: " + str(topic))

        return topic

    def predict_topic_df(self, df):
        # column names
        topicnames = ["Topic" + str(i) for i in range(self.best_lda_model.n_components)]

        df_index, data_lemmatized, df_index_empty = self.df_cleaning(df)

        if data_lemmatized:
            data_vectorized = self.vectorizer.transform(data_lemmatized)

            # Create Document - Topic Matrix
            lda_output = self.best_lda_model.transform(data_vectorized)

            # Make the pandas dataframe
            df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames,
                                             index=df_index.values.tolist())
        else:

            df_document_topic = pd.DataFrame(columns=topicnames,
                                             index=df_index.values.tolist())

        df_index_empty = df_index_empty.reindex(df_index_empty["t.id"].tolist()).drop("t.id", axis=1)

        df_document_topic = pd.merge(df_document_topic, df_index_empty, left_index=True, right_index=True,
                                     how='outer')

        df_document_topic['dominant_topic'] = df_document_topic.apply(self.get_topic, axis=1)

        # print (df_document_topic.head(10))

        df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
        df_topic_distribution.columns = ['Topic Num', 'Num Documents']
        # print (df_topic_distribution)

        return df_document_topic

    def get_topic(self, row):

        dominant_topic = randint(0, self.best_lda_model.n_components - 1)

        if np.argmax(row.values) != np.argmin(row.values):
            dominant_topic = np.argmax(row.values)

        return dominant_topic

    def save_model(self, file_name):
        lda_model_file = '../results/' + file_name + '.pkl'

        joblib.dump(self.best_lda_model, lda_model_file)

        print("---LDA model saved")

    def load_model(self, file_name):
        lda_model_file = '../results/' + file_name + '.pkl'

        self.best_lda_model = joblib.load(lda_model_file)

        print("---Model loaded")

    def model_validation(self, new=True, test_name='Test',
                         max_iterations=500, min_component=1, max_components=10,
                         step=5, sample_size=500):

        if new:
            df = self.neo_graph.run("MATCH (t:Tweet) RETURN t.id, t.text limit " + str(sample_size)).to_data_frame()

            df.to_csv(test_name + '_data.csv')
        else:
            df = pandas.read_csv(test_name + '_data.csv', low_memory=False)

        # Cleaning the data
        data_index, data_lemmatized, data_empty = self.df_cleaning(df)

        # features = self.vectorizer.fit_transform(data_lemmatized).toarray()

        features = self.tfidf.fit_transform(data_lemmatized).toarray()
        entries = []

        for num_cat in range(min_component, max_components + step, step):
            print(num_cat)
            lda_model = LatentDirichletAllocation(n_components=num_cat,  # Number of topics
                                                  max_iter=max_iterations,  # Max learning iterations
                                                  learning_method='online',
                                                  batch_size=128  # n docs in each learning iter
                                                  # evaluate_every=-1,  # compute perplexity every n iters, default: Don't
                                                  # n_jobs=-1,  # Use all available CPUs
                                                  )

            X_train, X_test, y_train, y_test = train_test_split(features, data_index, test_size=0.2, random_state=42)

            lda_model.fit_transform(X_train)
            perplexity = lda_model.perplexity(X_test)
            loglikelihood = lda_model.score(X_test)
            print(perplexity)
            print(loglikelihood)

            entries.append((num_cat, perplexity, loglikelihood))

        cv_df = pd.DataFrame(entries, columns=['Number of topics', 'Perplexity', 'Log likelihood'])

        cv_df.to_csv(test_name + '_perplexity_log.csv')

        seaborn.barplot(x='Number of topics', y='Perplexity', data=cv_df[['Number of topics', 'Perplexity']])

        plt.savefig(test_name + '_prin_perplexity.eps')
        plt.savefig(test_name + '_prin_perplexity.png')

        plt.show()

    def cleaning_data(self, df):
        stops = set(nltk.corpus.stopwords.words("english"))


        def removePunctuation(x):
            x = x.lower()
            x = re.sub(r'[^\x00-\x7f]', r' ', x)
            return re.sub("[" + string.punctuation + "]", " ", x)

        def removeStopwords(x):
            filtered_words = [word for word in x.split() if word not in stops]
            return " ".join(filtered_words)

        df['t.text'] = df['t.text'].apply(lambda x: re.split('http:\/\/.*', str(x))[0])
        df['t.text'] = df['t.text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
        reviews = [sent if type(sent) == str else "" for sent in df['t.text'].values]
        reviews = [removePunctuation(sent) for sent in reviews]
        reviews = [removeStopwords(sent) for sent in reviews]

        return reviews

    def generate_stats(self):

        df = self.neo_graph.run("MATCH (t:Tweet) RETURN t.id, t.text, t.topic limit 50").to_data_frame()

        # df = self.neo_graph.run("MATCH (a:User)-[f:FOLLOWS]->(n:User)-[p:POSTS]->(t:Tweet) WHERE n.id=\'21447363\' RETURN t.id, t.text, t.topic").to_data_frame()

        df['t.text_cleaned'] = self.cleaning_data(df)

        stopwords = set(STOPWORDS)
        wordcloud = WordCloud(background_color='white', stopwords=stopwords,
                              max_font_size=40, random_state=42)\
            .generate(str(df['t.text_cleaned'].values).replace('\n', '').replace('[', '').replace(']', '').replace('\'', '').replace('rt', ''))

        plt.figure(figsize=(10, 10))
        ax3 = plt.subplot2grid((2, 1), (0, 0))
        ax4 = plt.subplot2grid((2, 1), (1, 0))

        cat_hist = df.groupby('t.topic', as_index=False).count()

        cat_hist = cat_hist.sort_values(by='t.topic')
        sns.barplot(x=cat_hist['t.topic'].index, y=cat_hist['t.id'].values, ax=ax3)

        ax3.set_title("Topics", fontsize=16)
        ax4.set_title("Words Cloud", fontsize=16)
        ax4.imshow(wordcloud)
        ax4.axis('off')

        plt.savefig('../results/kdd_dt_wc.eps')
        plt.show()

        # --------------------------------------------------------------------

        document_lengths = numpy.array(df['t.text_cleaned'].str.split(' ').str.len())

        print("The average number of words in a document is: {}.".format(numpy.mean(document_lengths)))
        print("The minimum number of words in a document is: {}.".format(min(document_lengths)))
        print("The maximum number of words in a document is: {}.".format(max(document_lengths)))

        # --------------------------------------------------------------------

        stopwords.update(["rt", "en", "es", "que", "si", "nunca", "ufffff"])

        for id_topic in pandas.unique(df["t.topic"]):

            df_filtered = df[df['t.topic'] == id_topic]

            try:
                wordcloud = WordCloud(background_color='white', stopwords=stopwords, max_words=500,
                                      max_font_size=40, random_state=42) \
                    .generate(str(df_filtered['t.text_cleaned'].values).replace('\n', '').replace('[', '').replace(']', '').replace('\'', ''))

                wordcloud.to_file('../results/kdd_wc_'+ str(id_topic) + ".eps")
                plt.show()

            except:
                pass




    def generate_graphics(self, file_name_results):

        df_results = pandas.read_csv('../results/'+file_name_results+'.csv', low_memory=False)

        g = sns.barplot(x="Number of topics", y="Perplexity", hue="Model", data=df_results, errwidth=0)

        hatches = itertools.cycle(['/', '/', '\\', '-', '|', '*', 'o', 'O', '.'])
        c=-1
        hatch = next(hatches)
        for i, bar in enumerate(g.patches):
            c+=1
            if c % 3 == 0:
                hatch = next(hatches)
            bar.set_hatch(hatch)

        g.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fancybox=True, shadow=False)

        plt.savefig('../results/'+file_name_results + '.eps')

        plt.show()

