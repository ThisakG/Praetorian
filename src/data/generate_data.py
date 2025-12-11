import pandas as pd # type: ignore # lib used for storing data in a table and exporting CSVs
from datetime import datetime, timedelta #lib used for timestamp generation 
import random  #lib used for random data generation 

# function to create normal logins
def generate_normal_logins(n=1000):
    #creating 50 user IDs (user_001, user_002...etc)
    users = [f"user_{i:03d}" for i in range(1, 51)]

    #lists containing realistic locations and device types
    locations = ["Sri Lanka", "UK", "USA", "India", "Australia"]
    device_types = ["Windows", "Linux", "MacOS", "Android", "iOS"]

    #empty list to enter generated data
    rows = []

    #base starting time set to one week ago
    base_time = datetime.now() - timedelta(days=7)

    #generation of n rows 
    for _ in range(n):
        # selecting a random user using 'random'
        user = random.choice(users)

        # setting a random time within the last 7 days using 'random'
        timestamp = base_time + timedelta(seconds=random.randint(0, 604800))

        # setting IP addresses set to a private IP range with the last two octets randomized
        ip = f"192.168.{random.randint(0,10)}.{random.randint(1,255)}"

        #normal behaviour values
        success = 1 # normal login
        fails = 0 # no failed attempts

        # setting a random location using the list
        location = random.choice(locations)

        # setting a random session that is between 1-30 minutes
        session = random.randint(60, 1800)

        # appending row representing a single event
        rows.append([
            timestamp, user, ip,
            random.choice(device_types), # setting a random device type using the list
            success, fails, location, session,
            "normal" # labeling generated data as normal behaviour
        ])

    return rows

# function to generate anomalies
def generate_anomalies(n=50):
    #empty list to enter generated data
    anomalies = []
    #list with device types
    device_types = ["Windows", "Linux", "MacOS", "Android", "iOS"]
    # list with suspicious IPs
    attack_ips = ["66.102.0.1", "185.199.108.1", "101.44.1.98"]  

    base_time = datetime.now()

    #generation of n rows
    for _ in range(n):

        timestamp = base_time + timedelta(minutes=random.randint(1, 500))
        user = f"user_{random.randint(1, 50):03d}"

        # selecting an anomaly type randomly from a list
        anomaly_type = random.choice(["bruteforce", "impossible_travel", "weird_session", "suspicious_ip"])

        # if the selected type is bruteforce
        if anomaly_type == "bruteforce":
            anomalies.append([
                timestamp, user,
                f"203.0.{random.randint(0,255)}.{random.randint(0,255)}",
                random.choice(device_types),
                0, # login not successful
                random.randint(10, 25),
                "Unknown", # location when lots of failed attempts
                random.randint(5, 20),
                "anomalous" # label when very short session 
            ])

        # if the selected type is impossible travel 
        elif anomaly_type == "impossible_travel":
            anomalies.append([
                timestamp, user,
                f"122.1.{random.randint(0,255)}.{random.randint(0,255)}",
                random.choice(device_types),
                1,
                0,
                "Russia", # an unlikely location 
                random.randint(60, 200),
                "anomalous"
            ])

    # if the selected type is a weird session ( sessions that last several hours )
        elif anomaly_type == "weird_session":
            anomalies.append([
                timestamp, user,
                f"10.20.{random.randint(0,255)}.{random.randint(0,255)}",
                random.choice(device_types),
                1,
                0,
                "USA",
                random.randint(5000, 10000),
                "anomalous"
            ])

        # if the selected type is a suspicious IP address
        elif anomaly_type == "suspicious_ip":
            anomalies.append([
                timestamp, user,
                random.choice(attack_ips), # from list
                random.choice(device_types), # from list
                1,
                0,
                "Unknown",
                random.randint(20, 800),
                "anomalous"
            ])

    return anomalies

# function to save the dataset 
def save_dataset():

    # generating logs 
    normal = generate_normal_logins(1000)
    anomalous = generate_anomalies(50)

    # setting them into one dataset
    data = normal + anomalous

    # converting the log list to a data frame using pandas
    df = pd.DataFrame(data, columns=[
        "timestamp", "user_id", "ip", "device_type",
        "login_success", "failed_attempts",
        "location", "session_duration", "label"
    ])

    # shuffling rows so amomalies are not grouped together
    df = df.sample(frac=1).reset_index(drop=True)

    # saving as a .csv file
    df.to_csv("data/synthetic_logins.csv", index=False)
    print("Dataset saved to data/synthetic_logins.csv")

# Only run this when executing the script directly
if __name__ == "__main__":
    save_dataset()
