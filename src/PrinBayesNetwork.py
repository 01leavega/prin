import os
import jpype

import numpy
import pandas
from py2neo import Graph

import externallibs.data_frame_utils as dfu

MODELS_PATH = "Models/"


class PrinBayesNetwork:
    bayes_server = jpype.JPackage("com.bayesserver")
    neo_graph = Graph(host="", password="")


    network = None
    node_topics = []
    node_communities = []

    def __init__(self):

        # Connect to Bayes server through java virtual machine
        classpath = "externallibs/bayesserver-7.24.jar"

        if not jpype.isJVMStarted():
            jpype.startJVM(jpype.getDefaultJVMPath(),  "-ea", "-Xmx2048M", "-Djava.class.path=%s" % classpath, )

            License = jpype.JClass("com.bayesserver.License")
            License.validate("")

    def create_bayes_network(self, id_topics, id_communities, id_users, sources):

        self.network = None
        self.network = self.bayes_server.Network()
        self.node_topics = []
        self.node_communities = []

        # Add topics
        self.id_topics = id_topics

        # Add popularity
        self.add_popularity(sources)
        # self.add_similarity(sources)

        # Add communities
        for i in id_communities:
            self.add_community(i, id_topics)

        # Add users
        for u in id_users:
            self.add_users_source(u, sources)
            self.user_add_similarity(u, sources)
            self.add_user(u, id_topics, id_communities)

    def add_popularity(self, id_users_total):

        variable_users_popularity = self.bayes_server.Variable("g_user_popularity", id_users_total)

        # Create the node
        node_user_popularity = self.bayes_server.Node(variable_users_popularity)

        # Add the node to the network
        self.network.getNodes().add(node_user_popularity)

    def add_similarity(self, id_users_total):

        variable_users_similarity = self.bayes_server.Variable("g_user_similarity", id_users_total)

        # Create the node
        node_user_similarity = self.bayes_server.Node(variable_users_similarity)

        # Add the node to the network
        self.network.getNodes().add(node_user_similarity)

    def add_users_source(self, u, sources):
        # Create the variable with the specify states
        variable_users_source = self.bayes_server.Variable("u_" + u + "_user_source", sources)

        # Create the node
        node_user_source = self.bayes_server.Node(variable_users_source)

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

        # Add the node to the network
        self.network.getNodes().add(node_community_membership)

    def user_add_interest(self, id_user, topics):
        # Create the variable with the specify states
        variable_interest = self.bayes_server.Variable('u_' + id_user + "_interest", topics)

        # Create the node
        node_interest = self.bayes_server.Node(variable_interest)

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

        # Add the node to the network
        self.network.getNodes().add(node_community)

        link_parents = self.bayes_server.Link(self.network.getNodes().get('u_' + id_user + "_community_membership"),
                                              node_community)
        self.network.getLinks().add(link_parents)

    def user_add_structural_link(self, id_user):
        structural_states = ["true", "false"]

        variable_structural_link = self.bayes_server.Variable('u_' + id_user + "_structural_link", structural_states)

        node_structural_link = self.bayes_server.Node(variable_structural_link)
        self.network.getNodes().add(node_structural_link)

        link_parents = self.bayes_server.Link(self.network.getNodes().get('u_' + id_user + "_content_preference?"),
                                              node_structural_link)

        # link_source = self.bayes_server.Link(self.network.getNodes().get("u_"+id_user+"_user_source"),
        #                                       node_structural_link)

        link_popularity = self.bayes_server.Link(self.network.getNodes().get("g_user_popularity"),
                                                 node_structural_link)

        link_similarity = self.bayes_server.Link(self.network.getNodes().get('u_' + id_user + "_similarity"),
                                                 node_structural_link)

        self.network.getLinks().add(link_parents)
        # self.network.getLinks().add(link_source)
        self.network.getLinks().add(link_popularity)
        self.network.getLinks().add(link_similarity)

    def user_add_difussion_link(self, id_user):
        diffusion_states = ["true", "false"]

        variable_diffusion_link = self.bayes_server.Variable('u_' + id_user + "_diffusion_link", diffusion_states)

        node_diffusion = self.bayes_server.Node(variable_diffusion_link)
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

    def user_add_similarity(self, u, id_users_total):

        variable_user_similarity = self.bayes_server.Variable("u_" + u + "_similarity", id_users_total)

        # Create the node
        node_user_similarity = self.bayes_server.Node(variable_user_similarity)

        # Add the node to the network
        self.network.getNodes().add(node_user_similarity)

    # ------------Train-------------------------------------------------------------------------------------------------
    def train_network(self, network=None, datamodel=None, id_users=None, dm_similarity=None, dm_popularity=None, plot=False, max_iterations=200):
        # datamodel = datamodel.replace(numpy.nan, "", regex=True)

        # Set similarity
        if dm_similarity is not None:
            si_similarity = self.set_similarity(id_users=id_users, dm_similarity=dm_similarity)
        else:
            si_similarity = []

        if dm_popularity is not None:
            self.set_popularity(dm_popularity=dm_popularity)

        dt = dfu.to_data_table(datamodel)

        bayes_data = self.bayes_server.data
        bayes_inference = self.bayes_server.inference
        bayes_parameters = self.bayes_server.learning.parameters

        # We will use the RelevanceTree algorithm here, as it is optimized for parameter learning
        learning = bayes_parameters.ParameterLearning(self.network, bayes_inference.RelevanceTreeInferenceFactory())
        learning_options = bayes_parameters.ParameterLearningOptions()

        data_reader_command = bayes_data.DataTableDataReaderCommand(dt)

        reader_options = bayes_data.ReaderOptions()  # we do not have a case column in this example

        variables = self.network.getVariables()

        variable_references = []
        distribution_specification = []

        # for v in variables:
        for index in range(0, variables.size()):
            v = variables.get(index)

            if any(observed in v.getName() for observed in datamodel.columns):
                variable_references.append(
                    bayes_data.VariableReference(v, bayes_data.ColumnValueType.NAME, v.getName(),
                                                 bayes_data.StateNotFoundAction.MISSING_VALUE)
                )
            # Distribution to not learn
            if v.getName() != "g_user_popularity" and (v.getName() not in si_similarity):
                distribution_specification.append(
                    bayes_parameters.DistributionSpecification(self.network.getNodes().get(v.getName())))

        evidence_reader_command = bayes_data.DefaultEvidenceReaderCommand(
            data_reader_command,
            jpype.java.util.Arrays.asList(variable_references),
            reader_options)

        learning_options.setMaximumIterations(max_iterations)
        result = learning.learn(evidence_reader_command,
                                jpype.java.util.Arrays.asList(distribution_specification),
                                learning_options)
        if plot:
            print("\t Log likelihood = " + str(result.getLogLikelihood()))
            print("\t Iterations = " + str(result.getIterationCount()))
            print("\t Converged  = " + str(result.getConverged()))
            print("\t BIC  = " + str(result.getBIC()))

            parameter = self.bayes_server.ParameterCounter

            print("\t Parameter  = " + str(parameter.getParameterCount(self.network)))

        jpype.java.lang.System.gc()
        # jpype.shutdownJVM()

        return result

    def set_popularity(self, dm_popularity=None):
        # Setting the parameters

        table_g_user_popularity = self.network.getNodes().get("g_user_popularity").newDistribution().getTable()

        var_g_user_popularity = self.network.getVariables().get("g_user_popularity", True)

        for index, row in dm_popularity.iterrows():

            table_g_user_popularity.set(row['g_user_popularity'],
                                        [var_g_user_popularity.getStates().get(row['node.id'], True)])

        self.network.getNodes().get("g_user_popularity").setDistribution(table_g_user_popularity)

    def set_similarity(self, id_users=None, dm_similarity=None):

        list_user = list(dm_similarity['u.id'])
        si_similarity = []

        for u in id_users:

            if u in list_user:
                index_origin = dm_similarity[dm_similarity['u.id'] == u].index[0]

                dm_similarity["u_" + u + "_similarity"] = dm_similarity[index_origin]

                table_g_user_similarity = self.network.getNodes().get(
                    "u_" + u + "_similarity").newDistribution().getTable()

                var_g_user_similarity = self.network.getVariables().get("u_" + u + "_similarity", True)

                for index, row in dm_similarity.iterrows():
                    table_g_user_similarity.set(row["u_" + u + "_similarity"],
                                                [var_g_user_similarity.getStates().get(str(row['u.id']), True)])

                self.network.getNodes().get("u_" + u + "_similarity").setDistribution(table_g_user_similarity)

                # Adding the node with the parameters stablished
                node = "u_" + u + "_similarity"
                si_similarity.append(node)

        # print(si_similarity)

        return si_similarity

    def test_network(self, observed_datamodel=None, id_user=None, observed_variable='_diffusion_link'):
        observed_columns = observed_datamodel.columns

        if id_user is not None:
            col_y_real = 'u_'+id_user+observed_variable

        else:
            col_y_real = list(observed_datamodel.columns[observed_datamodel.columns.str.endswith(observed_variable)])[0]

        # print(col_y_real)
        if col_y_real in observed_columns:
            # Filter nan values in the col_y_real

            filtered_observed_datamodel = observed_datamodel.copy()
            filtered_observed_datamodel.dropna(subset=[col_y_real], inplace=True)

            y_real = filtered_observed_datamodel[col_y_real].replace(['true', 'false'], [1, 0])

            y_score = []

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

            for index, row in filtered_observed_datamodel.iterrows():
                for o_col in observed_columns:

                    if o_col != col_y_real:
                        try:
                            observed_variable = variables.get(o_col, True)
                            variable = observed_variable.getStates().get(row[o_col], True)
                            evidence.setState(variable)  # set discrete evidence
                        except:
                            # print("Not in states observed {} \t{}".format(o_col, row[o_col]))  # expected 0.5
                            pass

                # Query variables
                diffusionVariable = variables.get(col_y_real, True)
                queryDiffusion = self.bayes_server.Table(diffusionVariable)
                queryDistributions.add(bayes_server_inference.QueryDistribution(queryDiffusion))

                inference.query(queryOptions, queryOutput)

                diffusionTrue = diffusionVariable.getStates().get("true", True)
                diffusionFalse = diffusionVariable.getStates().get("false", True)

                diffusionValueTrue = queryDiffusion.get([diffusionTrue])
                # print("Diffusion = True \t{}".format(diffusionValueTrue))  # expected 0.5

                diffusionValueFalse = queryDiffusion.get([diffusionFalse])
                # print("Diffusion = False \t{}".format(diffusionValueFalse))  # expected 0.5

                y_score.append(diffusionValueTrue)

            y_score = numpy.asarray(y_score)
            y_real = numpy.asarray(y_real)

            return y_real, y_score

        return None, None

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
