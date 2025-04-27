from pyspark import SparkContext
from collections import defaultdict

# Initialize Spark context 
sc = SparkContext(appName="PeopleYouMightKnow")

# Load the input file as an RDD
data = sc.textFile("soc-LiveJournal1Adj.txt")

# Parse each line into (user, set of friends)
# Handle cases where users have no friends (empty friend list)
user_friends = data.map(lambda line: line.strip().split('\t')) \
    .filter(lambda x: len(x) >= 1) \
    .map(lambda x: (int(x[0]), set(map(int, x[1].split(','))) if len(x) == 2 and x[1] != '' else set()))

# Collect all user -> friends data into a dictionary and broadcast it to all workers
friend_dict = user_friends.collectAsMap()
broadcast_dict = sc.broadcast(friend_dict)

# Generate friend recommendations for a single user based on mutual friends
def generate_recommendations(user_friends_pair):
    user, friends = user_friends_pair
    mutuals = defaultdict(int)
    friend_dict = broadcast_dict.value

    # For each friend of the user
    for friend in friends:
        # Get their friends
        friends_of_friend = friend_dict.get(friend, set())
        for fof in friends_of_friend:
            # Recommend if not the user and not already a friend
            if fof != user and fof not in friends:
                mutuals[fof] += 1

    # Sort recommendations by number of mutual friends (descending), then by user ID (ascending) in case of tie
    sorted_candidates = sorted(mutuals.items(), key=lambda x: (-x[1], x[0]))
    recommendations = [str(candidate[0]) for candidate in sorted_candidates[:10]]
    
    return (user, recommendations)

# Apply the recommendation function to all users
recommendations = user_friends.map(generate_recommendations)

# Format output as a tab-separated string: <UserID> <TAB> <Comma-separated list of recommendations>
output_lines = recommendations.map(lambda x: f"{x[0]}\t{','.join(x[1])}").collect()

# Save all the results into a single .txt file
with open("recommendations_output.txt", "w") as f:
    for line in output_lines:
        f.write(line + "\n")

print("Results saved successfully to recommendations_output.txt!")

sc.stop()
