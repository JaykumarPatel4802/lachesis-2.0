import requests
import json
import urllib3
import pandas as pd

# Disable SSL certificate verification warning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Replace these values with your OpenWhisk configuration
OPENWHISK_API_URL = 'https://127.0.1.1:443'
AUTHORIZATION_KEY = 'MjNiYzQ2YjEtNzFmNi00ZWQ1LThjNTQtODE2YWE0ZjhjNTAyOjEyM3pPM3haQ0xyTU42djJCS0sxZFhZRnBYbFBrY2NPRnFtMTJDZEFzTWdSVTRWck5aOWx5R1ZDR3VNREdJd1A='
RESULTS_PER_PAGE = 100  # Adjust as needed

def get_activations(limit=10):
    headers = {
        'Authorization': f'Basic {AUTHORIZATION_KEY}',
        'Content-Type': 'application/json',
    }

    activations = []
    total_fetched = 0

    while total_fetched < limit:
        # Calculate the number of activations to fetch in this iteration
        remaining_to_fetch = limit - total_fetched
        fetch_count = min(remaining_to_fetch, RESULTS_PER_PAGE)

        # Calculate the offset for pagination
        offset = total_fetched

        # Make a GET request to fetch activations with SSL certificate verification disabled
        response = requests.get(
            f'{OPENWHISK_API_URL}/api/v1/namespaces/_/activations',
            headers=headers,
            params={'limit': fetch_count, 'skip': offset},
            verify=False  # Disable SSL certificate verification
        )

        if response.status_code == 200:
            activations.extend(response.json())
            total_fetched += fetch_count
        else:
            print(f'Failed to retrieve activations. Status code: {response.status_code}')
            break

    return activations

# def get_activations(limit=200):
#     headers = {
#         'Authorization': f'Basic {AUTHORIZATION_KEY}',
#         'Content-Type': 'application/json',
#     }

#     # You can adjust the limit to control how many activations you want to fetch
#     params = {
#         'limit': limit,
#     }

#     # Make a GET request to fetch activations
#     response = requests.get(
#         f'{OPENWHISK_API_URL}/api/v1/namespaces/_/activations',
#         headers=headers,
#         params=params,
#         verify=False
#     )

#     if response.status_code == 200:
#         activations = response.json()
#         return activations
#     else:
#         print(f'Failed to retrieve activations. Status code: {response.status_code}')
#         return None

if __name__ == '__main__':
    # Adjust the limit as needed
    activations = get_activations(limit=1200)
    
    if activations:
        # Initialize lists to store data
        activation_ids = []
        cpu_limits = []
        memory_limits = []
        wait_times = []
        init_times = []
        durations = []
        names = []
        start_times = []
        end_times = []
        status_codes = []

        for activation in activations:
            # Extract data from the activation JSON
            activation_id = activation['activationId']
            annotation = next((ann for ann in activation['annotations'] if ann['key'] == 'limits'), None)
            cpu_limit = annotation['value']['cpu'] if annotation else None
            memory_limit = annotation['value']['memory'] if annotation else None
            wait_time = next((ann['value'] for ann in activation['annotations'] if ann['key'] == 'waitTime'), 0)
            init_time = next((ann['value'] for ann in activation['annotations'] if ann['key'] == 'initTime'), 0)
            duration = activation['duration']
            name = activation['name'].split('_')[0]
            start_time = activation['start']
            end_time = activation['end']
            status_code = activation.get('statusCode', None)

            # Append extracted data to lists
            activation_ids.append(activation_id)
            cpu_limits.append(cpu_limit)
            memory_limits.append(memory_limit)
            wait_times.append(wait_time)
            init_times.append(init_time)
            durations.append(duration)
            names.append(name)
            start_times.append(start_time)
            end_times.append(end_time)
            status_codes.append(status_code)

        # Create a DataFrame from the lists
        data = {
            'activationId': activation_ids,
            'cpu': cpu_limits,
            'memory': memory_limits,
            'waitTime': wait_times,
            'initTime': init_times,
            'duration': durations,
            'name': names,
            'startTime': start_times,
            'endTime': end_times,
            'statusCode': status_codes,
        }

        df = pd.DataFrame(data)
        
        # Print the DataFrame
        print(df)