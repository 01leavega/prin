from py2neo import Graph, Node, Relationship
import numpy
import pandas
import os
import pickle
import joblib
import datetime

MODELS_PATH = "Models/"
PERIOD_BEGIN = "2018-08-01T00:00:00Z"
PERIOD_END = "2018-09-01T00:00:00Z"

class PrinDynamicDataModel:
    neo_graph = Graph(host="", password="")

    def generate_data_model(self, users):
        datamodel = pandas.DataFrame()


        for user in users:
            user_datamodel = self.data_model_user(id_user=user)

            if not user_datamodel.empty:
                if datamodel.empty:
                    datamodel = user_datamodel
                else:
                    datamodel = user_datamodel.merge(datamodel, left_on=['case','time'], right_on=['case','time'], how='outer')

        datamodel.to_csv(user+".csv", index=False)

        return datamodel

    def data_model_user(self, id_user):
        df_tweet = pandas.DataFrame()

        # Get all the tweets
        if PERIOD_BEGIN is None and PERIOD_END is None:
            df_tweet = pandas.DataFrame(self.neo_graph.data("MATCH (u:User)-[p:POSTS]->(t:Tweet) WHERE u.id={id} RETURN t.id, t.topic, t.created_at",
                                           {"id": str(id_user)}))
        else:
            df_tweet = pandas.DataFrame(self.neo_graph.data("MATCH (u:User)-[p:POSTS]->(t:Tweet) WHERE u.id={id} AND "
                                                            "DATETIME(t.datetime_created_at) > datetime({begin}) AND "
                                                            "DATETIME(t.datetime_created_at) < datetime({end})"
                                                            "RETURN t.id, t.topic, t.created_at",
                                                            {"id": str(id_user), "begin": str(PERIOD_BEGIN),
                                                             "end": str(PERIOD_END)}))


        if not df_tweet.empty:
            # for the temporal part

            df_tweet['t.created_at'] = pandas.to_datetime(df_tweet['t.created_at'])
            df_tweet = df_tweet.sort_values('t.created_at', ascending=True)

            df_tweet['case'] = pandas.to_datetime(df_tweet['t.created_at']).dt.month

            # Get all the retweets, with the sources and topic
            if PERIOD_BEGIN is None and PERIOD_END is None:
                df_retweet = pandas.DataFrame(self.neo_graph.data("MATCH (u:User)-[:POSTS]->(t:Tweet)-[:RETWEETS]->(o:Tweet)<-[:POSTS]-(x:User)<-[f:FOLLOWS]-(u:User)"
                                                          " WHERE u.id={id} "
                                                          "RETURN t.id as t_id, x.id as user_source, count(f)",
                                                   {"id": str(id_user)}))
            else:
                # Get all the retweets, with the sources and topic
                df_retweet = pandas.DataFrame(self.neo_graph.data(
                    "MATCH (u:User)-[:POSTS]->(t:Tweet)-[:RETWEETS]->(o:Tweet)<-[:POSTS]-(x:User)<-[f:FOLLOWS]-(u:User)"
                    " WHERE u.id={id} AND "
                    "DATETIME(t.datetime_created_at) > datetime({begin}) AND "
                    "DATETIME(t.datetime_created_at) < datetime({end})"
                    "RETURN t.id as t_id, x.id as user_source, count(f)",
                    {"id": str(id_user), "begin": str(PERIOD_BEGIN), "end": str(PERIOD_END)}))


            if not df_retweet.empty:
                # Since these are retweets there is a diffusion link of that tweet
                df_retweet['u_' + id_user + '_diffusion_link'] = 'true'

                # If the retweet come from a person that I follow
                df_retweet = df_retweet.replace({'count(f)': 1}, 'true')

                df_user = pandas.merge(df_tweet, df_retweet, left_on='t.id', right_on ='t_id', how='left')
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
                df_tweet[id_user + '_structural_link'] = 'false'
                df_tweet['u_'+ id_user + '_user_source'] = id_user

                df_tweet.rename(index=str, columns={'t.topic': 'u_' + id_user + '_message_topic_assigment'}, inplace=True)

                df_user = df_tweet
                df_user.drop(['t.id', 't.created_at'], axis=1, inplace=True)


            def create_sub_time(data):
                data['time'] = numpy.arange(data.shape[0]) + 1
                return data

            df_user = df_user.groupby(['case']).apply(create_sub_time)

            print(df_user.head(10))

            return df_user

        return df_tweet


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
