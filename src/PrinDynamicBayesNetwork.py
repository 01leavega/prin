import os

import numpy
import pandas
from jpype import *
from py2neo import Graph

import externallibs.data_frame_utils as dfu
from source.PrinBaseDataModel import PrinBaseDataModel

MODELS_PATH = "Models/"


class PrinDynamicBayesNetwork:
    bayes_server = JPackage("com.bayesserver")
    neo_graph = Graph(host="localhost", password="124578")

    network = None
    node_topics = []
    node_communities = []

    def __init__(self):

        # Connect to Bayes server through java virtual machine
        classpath = "externallibs/bayesserver-7.24.jar"

        startJVM(getDefaultJVMPath(), "-Djava.class.path=%s" % classpath)
        License = JClass("com.bayesserver.License")
        License.validate("a93a275b-8ff2-49bb-862a-5d67f85e97ed")

    def create_bayes_network(self, id_topics, id_communities, id_users, sources):

        self.network = self.bayes_server.Network()

        # Add topics
        self.id_topics = id_topics

        # Add popularity
        self.add_popularity(sources)
        self.add_similarity(sources)

        # Add communities
        for i in id_communities:
            self.add_community(i, id_topics)

        # Add users
        for u in id_users:
            self.add_users_source(u, sources)
            self.add_user(u, id_topics, id_communities)

    def add_popularity(self, id_users_total):

        variable_users_popularity = self.bayes_server.Variable("g_user_popularity", id_users_total)

        # Create the node
        node_user_popularity = self.bayes_server.Node(variable_users_popularity)
        # node_user_popularity.setTemporalType(self.bayes_server.TemporalType.TEMPORAL)

        # Add the node to the network
        self.network.getNodes().add(node_user_popularity)

    def add_similarity(self, id_users_total):

        variable_users_similarity = self.bayes_server.Variable("g_user_similarity", id_users_total)

        # Create the node
        node_user_similarity = self.bayes_server.Node(variable_users_similarity)
        node_user_similarity.setTemporalType(self.bayes_server.TemporalType.TEMPORAL)

        # Add the node to the network
        self.network.getNodes().add(node_user_similarity)

    def add_users_source(self, u, sources):
        # Create the variable with the specify states
        variable_users_source = self.bayes_server.Variable("u_" + u + "_user_source", sources)

        # Create the node
        node_user_source = self.bayes_server.Node(variable_users_source)
        node_user_source.setTemporalType(self.bayes_server.TemporalType.TEMPORAL)

        # Add the node to the network
        self.network.getNodes().add(node_user_source)

    def add_topic(self, id_topic, words):
        state_words = []

        # Add all the possible states
        for w in words:
            state_words.append(self.bayes_server.State(w))

        # Create the variable with the specify states
        variable_topic = self.bayes_server.Variable('t_' + id_topic + "_topic", state_words)

        # Create the node
        node_topic = self.bayes_server.Node(variable_topic)
        node_topic.setTemporalType(self.bayes_server.TemporalType.TEMPORAL)

        # Add the node to the network
        self.network.getNodes().add(node_topic)

        # Add the new topic to the list
        self.node_topics.append('t_' + id_topic + "_topic")

    def add_community(self, id_community, topics):
        state_topics = []

        # Add all the possible states
        for t in topics:
            state_topics.append(self.bayes_server.State(t))

        # Create the variable with the specify states
        variable_community = self.bayes_server.Variable('c_' + id_community + "_community_interest", state_topics)

        # Create the node
        node_community = self.bayes_server.Node(variable_community)
        node_community.setTemporalType(self.bayes_server.TemporalType.TEMPORAL)

        # Add the node to the network
        self.network.getNodes().add(node_community)

        # Add the node community to the list
        self.node_communities.append('c_' + id_community + "_community_interest")

    def add_user(self, id_user, topics, communities):
        self.user_add_community_membership(id_user, communities)
        # self.user_add_interest(id_user, topics)
        self.user_add_content_preference(id_user)
        self.user_message_add_community_assigment(id_user, communities)
        self.user_message_add_topic_assigment(id_user, topics)
        self.user_add_structural_link(id_user)
        self.user_add_difussion_link(id_user)

    def user_add_community_membership(self, id_user, communities):
        # Create the variable with the specify states
        variable_community_membership = self.bayes_server.Variable('u_' + id_user + "_community_membership",
                                                                   communities)

        # Create the node
        node_community_membership = self.bayes_server.Node(variable_community_membership)
        node_community_membership.setTemporalType(self.bayes_server.TemporalType.TEMPORAL)

        # Add the node to the network
        self.network.getNodes().add(node_community_membership)

    def user_add_interest(self, id_user, topics):
        # Create the variable with the specify states
        variable_interest = self.bayes_server.Variable('u_' + id_user + "_interest", topics)

        # Create the node
        node_interest = self.bayes_server.Node(variable_interest)
        node_interest.setTemporalType(self.bayes_server.TemporalType.TEMPORAL)

        # Add the node to the network
        self.network.getNodes().add(node_interest)

        link_parents_topic = self.bayes_server.Link(
            self.network.getNodes().get('u_' + id_user + "_message_topic_assigment"), node_interest)

        self.network.getLinks().add(link_parents_topic)

    def user_add_content_preference(self, id_user):
        # Create the variable with the specify states
        yes = self.bayes_server.State("yes")
        no = self.bayes_server.State("no")

        variable_content_preference = self.bayes_server.Variable('u_' + id_user + "_content_preference?", [yes, no])

        # Create the node
        node_content_preference = self.bayes_server.Node(variable_content_preference)
        node_content_preference.setTemporalType(self.bayes_server.TemporalType.TEMPORAL)

        # Add the node to the network
        self.network.getNodes().add(node_content_preference)

        link_source = self.bayes_server.Link(self.network.getNodes().get("u_" + id_user + "_user_source"),
                                             node_content_preference)

        self.network.getLinks().add(link_source)

    def user_message_add_topic_assigment(self, id_user, topics):
        # This will be calculated
        variable_topic = self.bayes_server.Variable('u_' + id_user + "_message_topic_assigment", topics)

        # Create the node
        node_topic = self.bayes_server.Node(variable_topic)
        node_topic.setTemporalType(self.bayes_server.TemporalType.TEMPORAL)

        # Add the node to the network
        self.network.getNodes().add(node_topic)

        for c in self.node_communities:
            link_parents = self.bayes_server.Link(self.network.getNodes().get(c),
                                                  node_topic)
            self.network.getLinks().add(link_parents)

        link_parents_community = self.bayes_server.Link(
            self.network.getNodes().get('u_' + id_user + "_message_community_assigment"),
            node_topic)
        self.network.getLinks().add(link_parents_community)

    def user_message_add_community_assigment(self, id_user, communities):
        variable_community = self.bayes_server.Variable('u_' + id_user + "_message_community_assigment", communities)

        # Create the node
        node_community = self.bayes_server.Node(variable_community)
        node_community.setTemporalType(self.bayes_server.TemporalType.TEMPORAL)

        # Add the node to the network
        self.network.getNodes().add(node_community)

        link_parents = self.bayes_server.Link(self.network.getNodes().get('u_' + id_user + "_community_membership"),
                                              node_community)
        self.network.getLinks().add(link_parents)

    def user_add_structural_link(self, id_user):
        structural_states = ["true", "false"]

        variable_structural_link = self.bayes_server.Variable('u_' + id_user + "_structural_link", structural_states)

        node_structural_link = self.bayes_server.Node(variable_structural_link)
        node_structural_link.setTemporalType(self.bayes_server.TemporalType.TEMPORAL)

        self.network.getNodes().add(node_structural_link)

        link_parents = self.bayes_server.Link(self.network.getNodes().get('u_' + id_user + "_content_preference?"),
                                              node_structural_link)

        # link_source = self.bayes_server.Link(self.network.getNodes().get("u_"+id_user+"_user_source"),
        #                                       node_structural_link)

        link_popularity = self.bayes_server.Link(self.network.getNodes().get("g_user_popularity"),
                                                 node_structural_link)

        link_similarity = self.bayes_server.Link(self.network.getNodes().get("g_user_similarity"),
                                                 node_structural_link)

        self.network.getLinks().add(link_parents)
        # self.network.getLinks().add(link_source)
        self.network.getLinks().add(link_popularity)
        self.network.getLinks().add(link_similarity)

    def user_add_difussion_link(self, id_user):
        diffusion_states = ["true", "false"]

        variable_diffusion_link = self.bayes_server.Variable('u_' + id_user + "_diffusion_link", diffusion_states)

        node_diffusion = self.bayes_server.Node(variable_diffusion_link)
        node_diffusion.setTemporalType(self.bayes_server.TemporalType.TEMPORAL)

        self.network.getNodes().add(node_diffusion)

        link_parents_structural_link = self.bayes_server.Link(
            self.network.getNodes().get('u_' + id_user + "_structural_link"),
            node_diffusion)

        link_parents_topic_assigment = self.bayes_server.Link(
            self.network.getNodes().get('u_' + id_user + "_message_topic_assigment"),
            node_diffusion)

        # link_parents_user_interest = self.bayes_server.Link(self.network.getNodes().get(id_user  + "_interest"),
        #                                                     node_diffusion)

        link_parents_user_source = self.bayes_server.Link(self.network.getNodes().get("u_" + id_user + "_user_source"),
                                                          node_diffusion)

        self.network.getLinks().add(link_parents_structural_link)
        self.network.getLinks().add(link_parents_topic_assigment)
        # self.network.getLinks().add(link_parents_user_interest)
        self.network.getLinks().add(link_parents_user_source)

    # ------------Train-------------------------------------------------------------------------------------------------
    def train_network(self, network=None, datamodel=None):

        if network is None:
            network = self.network

        dt = dfu.to_data_table(datamodel)

        bayes_data = self.bayes_server.data
        bayes_inference = self.bayes_server.inference
        bayes_parameters = self.bayes_server.learning.parameters

        # We will use the RelevanceTree algorithm here, as it is optimized for parameter learning
        learning = bayes_parameters.ParameterLearning(network, bayes_inference.RelevanceTreeInferenceFactory())
        learning_options = bayes_parameters.ParameterLearningOptions()

        temporal_data_reader_command = bayes_data.DataTableDataReaderCommand(dt)

        temporal_reader_options = bayes_data.TemporalReaderOptions("case", "time",
                                                                   bayes_data.TimeValueType.INDEX)  # we do not have a case column in this example

        variables = network.getVariables()

        variable_references = []
        distribution_specification = []

        for v in variables:

            if any(observed in v.getName() for observed in datamodel.columns):
                variable_references.append(
                    bayes_data.VariableReference(v, bayes_data.ColumnValueType.NAME, v.getName())
                )

            if v.getName() != "g_user_popularity":
                # if v.getName() != "g_user_popularity" and v.getName() != "g_user_similarity":
                distribution_specification.append(
                    bayes_parameters.DistributionSpecification(network.getNodes().get(v.getName())))

        evidence_reader_command = bayes_data.DefaultEvidenceReaderCommand(
            temporal_data_reader_command,
            java.util.Arrays.asList(variable_references),
            temporal_reader_options)

        result = learning.learn(evidence_reader_command,
                                java.util.Arrays.asList(distribution_specification),
                                learning_options)

        print("Log likelihood = " + str(result.getLogLikelihood()))
        print("Iterations = " + str(result.getIterationCount()))
        print("Converged  = " + str(result.getConverged()))

    def set_popularity(self, dm_popularity=None):
        # Setting the parameters

        table_g_user_popularity = self.network.getNodes().get("g_user_popularity").newDistribution().getTable()

        var_g_user_popularity = self.network.getVariables().get("g_user_popularity", True)

        for index, row in dm_popularity.iterrows():
            table_g_user_popularity.set(row['g_user_popularity'],
                                        [var_g_user_popularity.getStates().get(row['node.id'], True)])

        self.network.getNodes().get("g_user_popularity").setDistribution(table_g_user_popularity)

    def set_similarity(self, origin_id_user=None, dm_similarity=None):

        def get_similarity(data):
            data.reset_index(drop=True, inplace=True)

            index_origin = data[data['u.id'] == origin_id_user].index[0]

            origin_dm_similarity = pandas.DataFrame()
            origin_dm_similarity['g_user_similarity'] = data[index_origin].values
            origin_dm_similarity['u.id'] = data['u.id'].values
            origin_dm_similarity['case'] = data['case'].values

            return origin_dm_similarity

        origin_dm_similarity = dm_similarity.groupby(['case']).apply(get_similarity)

        table_g_user_similarity = self.network.getNodes().get("g_user_similarity").newDistribution().getTable()

        var_g_user_similarity = self.network.getVariables().get("g_user_similarity", True)

        for index, row in origin_dm_similarity.iterrows():
            table_g_user_similarity.set(row['g_user_similarity'],
                                        [self.bayes_server.newStateContex(
                                            var_g_user_similarity.getStates().get(str(row['u.id']), True),
                                            row['case'])])

        self.network.getNodes().get("g_user_similarity").setDistribution(table_g_user_similarity)

    def test_network(self):
        bayes_server_inference = self.bayes_server.inference

        factory = bayes_server_inference.RelevanceTreeInferenceFactory()

        inference = factory.createInferenceEngine(self.network)

        queryOptions = factory.createQueryOptions()
        queryOptions.setDecisionAlgorithm(bayes_server_inference.DecisionAlgorithm.SINGLE_POLICY_UPDATING)

        queryOutput = factory.createQueryOutput()

        queryDistributions = inference.getQueryDistributions()
        # Set some evidence
        evidence = inference.getEvidence()

        variables = self.network.getVariables()

        structuralVariable = variables.get("u_50393960_structural_link", True)
        structural = structuralVariable.getStates().get("false", True)
        evidence.setState(structural)  # set discrete evidence

        # Query variables
        diffusionVariable = variables.get("u_50393960_diffusion_link", True)
        queryDiffusion = self.bayes_server.Table(diffusionVariable)
        queryDistributions.add(bayes_server_inference.QueryDistribution(queryDiffusion))

        inference.query(queryOptions, queryOutput)

        diffusionTrue = diffusionVariable.getStates().get("true", True)
        diffusionFalse = diffusionVariable.getStates().get("false", True)

        diffusionValueTrue = queryDiffusion.get([diffusionTrue])
        print("Diffusion = True \t{}".format(diffusionValueTrue))  # expected 0.5

        diffusionValueFalse = queryDiffusion.get([diffusionFalse])
        print("Diffusion = False \t{}".format(diffusionValueFalse))  # expected 0.5

    def get_sources(self, id_users, datamodel):
        columns_sources = list(datamodel.columns[datamodel.columns.str.endswith('_user_source')])
        sources = numpy.unique(datamodel[columns_sources].values).tolist()

        id_users_total = list(set(id_users + sources))

        return id_users_total

    def save_object(self, object, file_name):
        file_name = os.path.join(MODELS_PATH, file_name)
        object.to_pickle(file_name)

    def load_object(self, file_name):
        file_name = os.path.join(MODELS_PATH, file_name)

        dataframe = pandas.read_pickle(file_name)

        return dataframe

    def save_network(self, file_name):
        file_name = os.path.join(MODELS_PATH, file_name)

        self.network.save(file_name + ".bayes")

        print("Network saved")

    def load_network(self, file_name):
        self.network = self.bayes_server.Network()

        file_name = os.path.join(MODELS_PATH, file_name)

        self.network.load(file_name + ".bayes")

        print("Network loaded")


if __name__ == '__main__':
    PERIOD_BEGIN = None  # "2018-08-01T00:00:00Z"
    PERIOD_END = None  # "2018-09-01T00:00:00Z"

    # TODO get the set of users "1450087165"

    id_users = ["50393960"]
    nodes_topics = ["0", "1", "2"]
    nodes_communities = ["0", "1"]

    dm_file_name = 'D_dm_1'
    s_dm_file_name = 'D_s_dm_1'
    p_dm_file_name = 'D_p_dm_1'
    bn_file_name = "D_bn_1"

    p = PrinBaseDataModel(period_begin=PERIOD_BEGIN, period_end=PERIOD_END, temporal=True)

    datamodel = p.generate_data_model(id_users)

    sources = p.get_sources(id_users, datamodel)

    s_datamodel = p.get_data_model_similarity(id_users=sources, nodes_topics=nodes_topics)
    p_datamodel = p.get_data_model_popularity(id_users=sources)

    p.save_csv(object=datamodel, file_name=dm_file_name)
    p.save_csv(object=s_datamodel, file_name=s_dm_file_name)
    p.save_csv(object=p_datamodel, file_name=p_dm_file_name)

    c = PrinDynamicBayesNetwork()

    c.create_bayes_network(nodes_topics, nodes_communities, id_users, sources)

    # c.set_similarity(origin_id_user="50393960", dm_similarity=s_datamodel)

    c.set_popularity(dm_popularity=p_datamodel)

    # network = c.load_network(bn_file_name)
    #
    c.train_network(network=None, datamodel=datamodel)

    c.save_network(bn_file_name)
    #
    # # name_network = "bn_50393960_trained"
    # # c.load_network(name_network)
    #
    # c.test_network()
