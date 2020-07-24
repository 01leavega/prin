import os

import numpy
import pandas
from py2neo import Graph
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

MODELS_PATH = "Models/"


class PrinBaseDataModel:

    def __init__(self, period_begin=None, period_end=None, temporal=False):
        self.period_begin = period_begin
        self.period_end = period_end
        self.neo_graph = Graph(host="", password="")
        self.temporal = temporal

    def generate_data_model(self, id_users):
        datamodel = pandas.DataFrame()

        for user in id_users:
            user_datamodel = self.get_data_model_user(id_user=user)

            if not user_datamodel.empty:
                user_datamodel = pandas.DataFrame(user_datamodel, dtype='str')

                if datamodel.empty:
                    datamodel = user_datamodel
                else:
                    datamodel = user_datamodel.merge(datamodel, left_on=['case', 'time'], right_on=['case', 'time'],
                                                     how='outer')

        if not self.temporal:
            if not datamodel.empty:
                datamodel.drop(columns=['case', 'time'], inplace=True)
            else:
                print("ERROR: Empty datamodel for {}".format(str(id_users)))

        return datamodel

    def get_data_model_user(self, id_user):

        # Get all the tweets
        if self.period_begin is None and self.period_end is None:
            df_tweet = self.neo_graph.run(
                "MATCH (u:User)-[p:POSTS]->(t:Tweet) WHERE u.id={id} RETURN t.id, t.topic, t.created_at",
                {"id": str(id_user)}).to_data_frame()

        else:
            df_tweet = self.neo_graph.run("MATCH (u:User)-[p:POSTS]->(t:Tweet) WHERE u.id={id} AND "
                                          "DATETIME(t.datetime_created_at) >= datetime({begin}) AND "
                                          "DATETIME(t.datetime_created_at) < datetime({end})"
                                          "RETURN t.id, t.topic, t.created_at",
                                          {"id": str(id_user), "begin": str(self.period_begin),
                                           "end": str(self.period_end)}).to_data_frame()

        if not df_tweet.empty:
            # for the temporal part

            df_tweet['t.created_at'] = pandas.to_datetime(df_tweet['t.created_at'])
            df_tweet = df_tweet.sort_values('t.created_at', ascending=True)

            df_tweet['case'] = pandas.to_datetime(df_tweet['t.created_at']).dt.month

            # Get all the retweets, with the sources and topic
            if self.period_begin is None and self.period_begin is None:
                df_retweet = self.neo_graph.run(
                    "MATCH (u:User)-[:POSTS]->(t:Tweet)-[:RETWEETS]->(o:Tweet)<-[:POSTS]-(x:User)<-[f:FOLLOWS]-(u:User)"
                    " WHERE u.id={id} "
                    "RETURN t.id as t_id, x.id as user_source, count(f)",
                    {"id": str(id_user)}).to_data_frame()
            else:
                # Get all the retweets, with the sources and topic
                df_retweet = self.neo_graph.run(
                    "MATCH (u:User)-[:POSTS]->(t:Tweet)-[:RETWEETS]->(o:Tweet)<-[:POSTS]-(x:User)<-[f:FOLLOWS]-(u:User)"
                    " WHERE u.id={id} AND "
                    "DATETIME(t.datetime_created_at) > datetime({begin}) AND "
                    "DATETIME(t.datetime_created_at) < datetime({end})"
                    "RETURN t.id as t_id, x.id as user_source, count(f)",
                    {"id": str(id_user), "begin": str(self.period_begin), "end": str(self.period_end)}).to_data_frame()

            if not df_retweet.empty:
                # Since these are retweets there is a diffusion link of that tweet
                df_retweet['u_' + id_user + '_diffusion_link'] = 'true'

                # If the retweet come from a person that I follow
                df_retweet['count(f)'] = numpy.where(df_retweet['count(f)'] > 0, 'true', 'false')

                df_user = pandas.merge(df_tweet, df_retweet, left_on='t.id', right_on='t_id', how='left')
                df_user.drop(['t_id', 't.id', 't.created_at'], axis=1, inplace=True)

                new_name_columns = {
                    'count(f)': 'u_' + id_user + '_structural_link',
                    't.topic': 'u_' + id_user + '_message_topic_assigment',
                    'user_source': 'u_' + id_user + '_user_source'
                }

                df_user.rename(index=str, columns=new_name_columns, inplace=True)

                fill_values = {
                    'u_' + id_user + '_structural_link': 'false',
                    'u_' + id_user + '_diffusion_link': 'false',
                    'u_' + id_user + '_user_source': id_user
                }

                df_user = df_user.fillna(value=fill_values)

            else:

                # Since these are retweets there is a diffusion link of that tweet
                df_tweet['u_' + id_user + '_diffusion_link'] = 'false'
                df_tweet['u_' + id_user + '_structural_link'] = 'false'
                df_tweet['u_' + id_user + '_user_source'] = id_user

                df_tweet.rename(index=str, columns={'t.topic': 'u_' + id_user + '_message_topic_assigment'},
                                inplace=True)

                df_user = df_tweet
                df_user.drop(['t.id', 't.created_at'], axis=1, inplace=True)

            def create_sub_time(data):
                data['time'] = numpy.arange(data.shape[0]) + 1
                return data

            df_user = df_user.groupby(['case']).apply(create_sub_time)

            return df_user

        return df_tweet

    def get_data_model_similarity(self, id_users=None, nodes_topics=[]):

        if self.temporal:
            query_temporal = 'DATETIME (t.datetime_created_at).month AS case,'
            query_month = ', case'
        else:
            query_temporal = ''
            query_month = ''

        if id_users is None:
            if self.period_end is None and self.period_begin is None:
                query = 'MATCH (u:User)- [:POSTS]->(t:Tweet) ' \
                        ' \n WITH u, ' + query_temporal + 'COLLECT(t) AS ts \n WITH u' + query_month

            else:
                query = 'MATCH (u:User)- [:POSTS]->(t:Tweet) ' \
                        'WHERE  ' \
                        'DATETIME(t.datetime_created_at) >= datetime({begin}) AND ' \
                        'DATETIME(t.datetime_created_at) < datetime({end})' \
                        ' \n WITH u, ' + query_temporal + 'COLLECT(t) AS ts \n WITH u' + query_month

        else:
            if len(id_users) > 1:
                id_users_str = "\",\"".join( map(str, id_users))  
                id_users_str = "[\"" + id_users_str + "\"]"
            else:
                id_users_str = str(id_users)

            if self.period_end is None and self.period_begin is None:
                query = 'WITH ' + id_users_str + ' AS arr ' \
                                                       'MATCH (u:User)- [:POSTS]->(t:Tweet) ' \
                                                       'WHERE u.id in arr ' \
                                                       'WITH u, ' + query_temporal + 'COLLECT(t) AS ts WITH u' + query_month

            else:
                query = 'WITH ' + id_users_str + ' AS arr ' \
                                                       'MATCH (u:User)- [:POSTS]->(t:Tweet) ' \
                                                       'WHERE u.id in arr AND ' \
                                                       'DATETIME(t.datetime_created_at) >= datetime({begin}) AND ' \
                                                       'DATETIME(t.datetime_created_at) < datetime({end}) \n' \
                                                       'WITH u, ' + query_temporal + 'COLLECT(t) AS ts WITH u' + query_month

        for t in nodes_topics:
            query = query + ', FILTER (x IN ts WHERE x.topic = ' + str(t) + ') AS T' + str(t)

        query = query + ' RETURN u.id ' + query_month

        for t in nodes_topics:
            query = query + ', LENGTH(T' + str(t) + ')'

        if self.period_end is None and self.period_begin is None:
            df_similarity = self.neo_graph.run(query).to_data_frame()

        else:
            df_similarity = self.neo_graph.run(query, {"begin": str(self.period_begin),
                                                       "end": str(self.period_end)}).to_data_frame()

        if self.temporal:

            def get_similarity(data):
                np_similarity = numpy.array(data.drop(columns=["u.id", "case"], axis=1))

                A_sparse = sparse.csr_matrix(np_similarity)

                similarities = cosine_similarity(A_sparse)

                # also can output sparse matrices
                # similarities_sparse = cosine_similarity(A_sparse, dense_output=False)
                # print('pairwise sparse output:\n {}\n'.format(similarities_sparse))

                pd_cosine = pandas.DataFrame(similarities)

                zero_columns = pd_cosine.columns[(pd_cosine == 0).all()]

                for c in zero_columns:
                    pd_cosine[c] = 1

                pd_cosine = pd_cosine.div(pd_cosine.sum(axis=0))

                pd_cosine["u.id"] = data["u.id"].values
                pd_cosine["case"] = data["case"].values

                return pd_cosine

            pd_cosine = df_similarity.groupby(['case']).apply(get_similarity)

        else:
            np_similarity = numpy.array(df_similarity.drop(columns=["u.id"], axis=1))

            A_sparse = sparse.csr_matrix(np_similarity)

            similarities = cosine_similarity(A_sparse)

            # also can output sparse matrices
            # similarities_sparse = cosine_similarity(A_sparse, dense_output=False)
            # print('pairwise sparse output:\n {}\n'.format(similarities_sparse))

            pd_cosine = pandas.DataFrame(similarities)

            zero_columns = pd_cosine.columns[(pd_cosine == 0).all()]

            for c in zero_columns:
                pd_cosine[c] = 1

            pd_cosine = pd_cosine.div(pd_cosine.sum(axis=0))

            pd_cosine["u.id"] = df_similarity["u.id"]

        return pd_cosine

    def get_data_model_popularity(self, id_users=None):
        df_popularity = pandas.DataFrame()

        if id_users is None:
            df_popularity = self.neo_graph.run(
                "CALL algo.pageRank.stream('User', 'FOLLOWS', {iterations:5}) YIELD node, score WITH * ORDER BY score DESC RETURN node.id, score score as g_user_popularity").to_data_frame()

        else:
            # Add just the users in the subgrap
            if len(id_users) > 1:
                id_users_str = "\",\"".join(
                    map(str, id_users))  
                id_users_str = "[\"" + id_users_str + "\"]"
            else:
                id_users_str = str(id_users)

            query = 'CALL algo.pageRank.stream(' \
                    '\'WITH ' + id_users_str + ' AS arr MATCH  (u:User) WHERE u.id IN arr RETURN id(u) as id\',' \
                                                     '\'MATCH (u1:User)-[:FOLLOWS]->(u2:User) RETURN id(u1) as source, id(u2) as target\',' \
                                                     '{graph: \'cypher\', iterations: 20}) YIELD nodeId, score \n' \
                                                     'MATCH(node) WHERE id(node) = nodeId RETURN node.id, score as g_user_popularity'

            df_popularity = self.neo_graph.run(query).to_data_frame()

        df_popularity["g_user_popularity"] = df_popularity["g_user_popularity"].div(
            df_popularity["g_user_popularity"].sum(axis=0))

        return df_popularity

    def get_sources(self, id_users, datamodel):
        columns_sources = list(datamodel.columns[datamodel.columns.str.endswith('_user_source')])
        sources = numpy.unique(datamodel[columns_sources].dropna().values).tolist()

        id_users_total = list(set(id_users + sources))

        return id_users_total

    def save_object(self, object, file_name):
        file_name = os.path.join(MODELS_PATH, file_name)
        object.to_pickle(file_name)

    def load_object(self, file_name):
        file_name = os.path.join(MODELS_PATH, file_name)

        dataframe = pandas.read_pickle(file_name)

        return dataframe

    def save_csv(self, object, file_name):
        file_name = os.path.join(MODELS_PATH, file_name)

        object.to_csv(file_name + ".csv", index=False)

        print('Saved ' + file_name)
