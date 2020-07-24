
# Probabilistic Reasoning System for Social Influence Analysis

## Abstract
Social Influence Analysis is a vast research field that has attracted research interest in many areas. This is because social influence is an invisible, complex and subtle phenomenon that governs social dynamics and user behaviors in social networks. Modeling and quantify social influence can provide the following advantages: understanding of social behaviors of people from the angle of sociology, experts finding, link prediction, promoting communication and dissemination of political, economic, and cultural activities as well as in other fields. Although, much effort has been done in this field, a fundamental problem is to characterize the social influence process as close to reality. Solving this problem is non--trivial, it is challenging in the following aspects: (1) it is still not clear how to characterize social influence, due to the complexity of abstract concepts from social science and noisy human behavior, (2) it is also not clear what factors should be considered to construct social influence and the ways to properly integrate those factors to evaluate each user's influence, (3) it is hard to validate a model due to the lack of ground truth about social influence and commonly accepted standard metrics. This thesis presents a novel probabilistic graphical model to characterizes the social influence process by its observed effect inside an online social network. The model leverages temporal heterogeneous information and textual content associated with each user in the network to mine topic-level individual influences strength. The model is implemented in a new Probabilistic Reasoning system for social INfluence Analysis named PRIN. 

# Folders and files
- data
	- Dump with the test data from Neo4j DB 
- src
	- PrinBaseDataModel.py
	- PrinBayesNetwork.py
		- Additional parameters to specify:
			- License.validate("")
	- PrinDynamicBayesNetwork.py
	- PrinDynamicDataModel.py
	- PrinLDAModel.py
	- TwitterDataCrawler.py
		- Additional parameters to specify:
			- Neo4j configuration
				- neo_graph = Graph(host="", password="")
			- Twitter API credentials
				- consumer_key = ""
				- consumer_secret = ""
				- access_token = ""
				- access_token_secret = ""
	- externallibs.zip
