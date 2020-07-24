import time
import twitter
from py2neo import Graph, Node, Relationship
import logging
import random
import queue
import datetime

LEVEL = 3
SEED = '50393960'  # BillGates
# SEED =  '21447363' # KatyPerry
# SEED = '155659213' #Cristiano
# SEED = '25073877'  # Donal Trump


POSTS_COUNT = 300
RETWEETERS_COUNT = 200
FAVORITE_COUNT = 20
FOLLOWERS_COUNT= 200
FRIEND_COUNT = 200


class TwitterData:
    twitter_api = twitter.Api()
    neo_graph = Graph(host="", password="")
    user_queue = queue.Queue()

    def __init__(self, consumer_key, consumer_secret, access_token, access_token_secret):

        logging.basicConfig(filename='twitterCrawler_'+str(datetime.datetime.now())+'.log', level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S', filemode='w')

        start_time = time.time()
        print('Start Crawler: ' + time.strftime("%d %m %Y %H %M %S"))
        logging.info('Start Crawler: ' + time.strftime("%d %m %Y %H %M %S"))


        # Twitter connection
        try:
            self.twitter_api = twitter.Api(consumer_key=consumer_key,
                                           consumer_secret=consumer_secret,
                                           access_token_key=access_token,
                                           access_token_secret=access_token_secret,
                                           sleep_on_rate_limit=True)

            print("Successful connection with Twitter API")

        except:
            print("Error connection with Twitter API")

        # Neo4j connection
        try:
            # self.neo_graph.delete_all()
            print("Successful connection with Neo4j")
        except:
            print("Error connection with Neo4j")


        self.get_graph_from(user_twitter_id=SEED, level=LEVEL)

        logging.info("--- %s seconds ---" % (time.time() - start_time))
        logging.info('End: ' + time.strftime("%d %m %Y %H %M %S"))
        print("--- %s seconds ---" % (time.time() - start_time))
        print('End: ' + time.strftime("%d %m %Y %H %M %S"))

    def get_graph_from(self, user_twitter_id, level):
        # Recursive
        # self.add_user(user_id, level)

        # Iterative
        self.user_queue.put(user_twitter_id)

        while not self.user_queue.empty():
            user_id =self.user_queue.get()

            posts = self.get_posts(user_id)

            if posts.__len__() > 100:  # TODO 20
                user = self.twitter_api.GetUser(user_id=user_id, include_entities=True, return_json=False)

                # Insert node User
                node_user = self.create_node_user(user)

                for p in posts:
                    # Insert Tweet node
                    node_tweet = self.create_node_tweet(p)

                    # Insert POST relationship
                    self.neo_graph.merge(Relationship(node_user, "POSTS", node_tweet))

                    # Evaluate post
                    self.check_tweet_entities(node_user, node_tweet, p, level)

                    # Check all the reetweeters of the tweet
                    for r in self.get_tweet_retweeters(p.id):

                        self.user_queue.put(r)

                # Checks favorites Tweets
                favorites = self.get_tweet_favorites(user_id)
                for f in favorites:
                    node_favorite_tweet = self.create_node_tweet(f)
                    # Insert LIKES relationship
                    self.neo_graph.merge(Relationship(node_user, "LIKES", node_favorite_tweet))

                    self.check_tweet_entities(node_user, node_favorite_tweet, f, level)

                    favorite_user_tweet = self.twitter_api.GetUser(user_id=f.user.id, include_entities=True,
                                                                   return_json=False)
                    node_favorite_user_tweet = self.create_node_user(favorite_user_tweet)

                    # Insert FOLLOWS relationship
                    self.neo_graph.merge(Relationship(node_user, "FOLLOWS", node_favorite_user_tweet))

                    self.user_queue.put(f.user.id)

                # Check users who I follow
                friends = self.get_friends(user_id)
                for f in friends:
                    friend = self.twitter_api.GetUser(user_id=f, include_entities=True, return_json=False)
                    node_friend = self.create_node_user(friend)

                    self.neo_graph.merge(Relationship(node_user, "FOLLOWS", node_friend))

                    self.user_queue.put(f)


                # Check users who follow me
                followers = self.get_followers(user_id)
                for f in followers:
                    follower = self.twitter_api.GetUser(user_id=f.id, include_entities=True, return_json=False)
                    node_follower = self.create_node_user(follower)

                    self.neo_graph.merge(Relationship(node_follower, "FOLLOWS", node_user))

                    self.user_queue.put(f.id)

    def add_user(self, user_id, level):
        logging.info(str(user_id) + " - " + str(level))

        if level > 0:
            # Check if is an active user
            posts = self.get_posts(user_id)

            if posts.__len__() > 20:  # TODO 20
                user = self.twitter_api.GetUser(user_id=user_id, include_entities=True, return_json=False)

                # Insert node User
                node_user = self.create_node_user(user)

                for p in posts:
                    # Insert Tweet node
                    node_tweet = self.create_node_tweet(p)

                    # Insert POST relationship
                    self.neo_graph.merge(Relationship(node_user, "POSTS", node_tweet))

                    # Evaluate post
                    self.check_tweet_entities(node_user, node_tweet, p, level)

                    # Check all the reetweeters of the tweet
                    for r in self.get_tweet_retweeters(p.id):
                        level = random.randint(1, 2)
                        self.add_user(r, level)

                # Checks favorites Tweets
                favorites = self.get_tweet_favorites(user_id)
                for f in favorites:
                    node_favorite_tweet = self.create_node_tweet(f)
                    # Insert LIKES relationship
                    self.neo_graph.merge(Relationship(node_user, "LIKES", node_favorite_tweet))

                    self.check_tweet_entities(node_user, node_favorite_tweet, f, level)

                    favorite_user_tweet = self.twitter_api.GetUser(user_id=f.user.id, include_entities=True,
                                                                   return_json=False)
                    node_favorite_user_tweet = self.create_node_user(favorite_user_tweet)

                    # Insert FOLLOWS relationship
                    self.neo_graph.merge(Relationship(node_user, "FOLLOWS", node_favorite_user_tweet))

                    self.add_user(f.user.id, level - 1)

                # Check users who I follow
                friends = self.get_friends(user_id)
                for f in friends:
                    friend = self.twitter_api.GetUser(user_id=f, include_entities=True, return_json=False)
                    node_friend = self.create_node_user(friend)

                    self.neo_graph.merge(Relationship(node_user, "FOLLOWS", node_friend))
                    self.add_user(f, level - 1)

                # Check users who follow me
                followers = self.get_followers(user_id)
                for f in followers:
                    follower = self.twitter_api.GetUser(user_id=f.id, include_entities=True, return_json=False)
                    node_follower = self.create_node_user(follower)

                    self.neo_graph.merge(Relationship(node_follower, "FOLLOWS", node_user))
                    # level = random.randint(1, 3)
                    self.add_user(f.id, level -1)

    def check_tweet_entities(self, node_user, node_tweet, tweet, level):
        # Check if the Tweet has hashtags
        if tweet.hashtags.__len__() > 0:
            for h in tweet.hashtags:
                node_hashtag = self.create_node_hashtag(h)
                # Insert HAS_HASHTAG relationship
                self.neo_graph.merge(Relationship(node_tweet, "HAS_HASHTAG", node_hashtag))

        # Check if the Tweet has url
        if tweet.urls.__len__() > 0:
            for u in tweet.urls:
                # Insert Url node
                node_url = self.create_node_url(u)
                # Insert HAS_URL relationship
                self.neo_graph.merge(Relationship(node_tweet, "HAS_URL", node_url))

        # Check if the Tweet is a retweet of other
        #if tweet.retweeted == True:
            # Check if the tweeet have been created before
        if tweet.retweeted_status:
            print ("retweeted")
            node_tweet_original = self.create_node_tweet(tweet.retweeted_status)

            # Check if the user is the graph already
            twitter_user_tweet_original = self.twitter_api.GetUser(user_id=tweet.retweeted_status.user.id,
                                                                   include_entities=True, return_json=False)
            node_user_tweet_original = self.create_node_user(twitter_user_tweet_original)

            # Insert FOLLOWS relationship
            self.neo_graph.merge(Relationship(node_user, "FOLLOWS", node_user_tweet_original))

            # Insert POST relationship
            self.neo_graph.merge(Relationship(node_user_tweet_original, "POSTS", node_tweet_original))

            # Insert RETWEETS relationship
            self.neo_graph.merge(Relationship(node_tweet, "RETWEETS", node_tweet_original))

            # level = random.randint(1, 3)
            # Recursive self.add_user(tweet.retweeted_status.user.id, level -1)
            self.user_queue.put(tweet.retweeted_status.user.id)

    def get_posts(self, user_id):
        return self.twitter_api.GetUserTimeline(user_id=user_id, trim_user=True, count=POSTS_COUNT)

    def get_tweet_information(self, tweet_id):
        return self.twitter_api.GetStatus(status_id=tweet_id, trim_user=False)

    def get_tweet_retweeters(self, tweet_id):
        return self.twitter_api.GetRetweeters(status_id=tweet_id, count=RETWEETERS_COUNT)

    def get_tweet_favorites(self, user_id):
        return self.twitter_api.GetFavorites(user_id=user_id, count=FAVORITE_COUNT)

    def get_followers(self, user_id):
        return self.twitter_api.GetFollowers(user_id=user_id, total_count=FOLLOWERS_COUNT)

    def get_friends(self, user_id):
        return self.twitter_api.GetFriendIDs(user_id=user_id, total_count=FRIEND_COUNT)

    def create_node_user(self, user):
        node_user = Node("User",
                         id=user.id_str,
                         name=user.name,
                         screen_name=user.screen_name,
                         description=user.description,
                         protected=user.protected,
                         created_at=user.created_at,
                         popularity=0)
        self.neo_graph.merge(node_user)

        return node_user

    def create_node_tweet(self, tweet):
        # TODO:Validate new created_at
        node_tweet = Node("Tweet",
                          id=tweet.id_str,
                          created_at=tweet.created_at,
                          datetime_created_at=datetime.datetime.strptime(tweet.created_at, "%a %b %d %H:%M:%S +0000 %Y").isoformat()+'Z',
                          text=tweet.text)
        # Insert node Tweet
        self.neo_graph.merge(node_tweet)

        return node_tweet

    def create_node_hashtag(self, hashtags):
        node_hashtag = Node("Hashtag",
                            text=hashtags.text)

        # Insert node Hashtag
        self.neo_graph.merge(node_hashtag)

        return node_hashtag

    def create_node_url(self, url):
        node_url = Node("Url",
                        url=url.url)

        # Insert node URL
        self.neo_graph.merge(node_url)

        return node_url


if __name__ == '__main__':
    consumer_key = ""
    consumer_secret = ""
    access_token = ""
    access_token_secret = ""

    c = TwitterData(consumer_key, consumer_secret, access_token, access_token_secret)


