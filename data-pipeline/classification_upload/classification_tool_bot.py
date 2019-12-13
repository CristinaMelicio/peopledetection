import sched, time
import requests
import json
import os
from requests.exceptions import HTTPError
import pandas as pd
import datetime

SEQUENCE_DIR = "/media/nas/PeopleDetection/up_realsense_frames/Data_Seq/AVI"
SEQUENCE_RECORDS = "/media/nas/PeopleDetection/up_realsense_frames/Data_Seq/sequence_labeling_record.csv"
SERVER_URL = "http://visionclassificationtool.inov.pt"
SEQUENCE_LABELS = '/media/nas/PeopleDetection/up_realsense_frames/Data_Seq/labels/Sharp_Dataset'
MAX_UPLOADS = 10
MIN_SIZE = 50 * 1024
MIN_SIZE_MPEG = 600 * 1024
MPEG_START = 73
INTERVAL_LD = 10 * 60
INTERVAL_OPS = 5
CLASSES = [
    "Person", 
    "Handcart", 
    "Phone", 
    "Bag", 
    "Documents", 
    "Helmet", 
    "Card", 
    "Wheelchair"]

def login():
    r = requests.post(SERVER_URL+'/api/users/login', json={
        'email': 'datasetbot@inov.pt',
        'password': 'inov'
    })
    r.raise_for_status()
    return r.json()["token"]

def upload_sequence(sequence, session):
    print('Uploading sequence '+sequence['name'])
    filename = SEQUENCE_DIR+'/'+sequence['name']+'.mp4'
    if os.path.getsize(filename) < MIN_SIZE:
        return True
    seq_number = int(sequence['name'].split('_')[0])
    if seq_number >= 73 and os.path.getsize(filename) < MIN_SIZE_MPEG:
        return True
    try:
        dataset_res = session.post(SERVER_URL+'/api/dataset', json={
            'name': sequence['name']
        })
        dataset_res.raise_for_status()
        dataset = dataset_res.json()
        
        files = {}
        files[dataset['_id']+'_RGB'] = (filename, open(filename, 'rb'), 'video/mp4')
        
        upload_res = session.post(SERVER_URL+'/api/upload/dataset', files=files)
        upload_res.raise_for_status()
        
        job_res = session.post(SERVER_URL+'/api/job', json={
            'dataset': dataset['_id'],
            'classes': CLASSES,
            'type': 'Tracking',
            'priority': 1,
            'channel': 'RGB'
        })
        job_res.raise_for_status()
        return True
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
        return False

def upload_new_sequences(scheduler): 
    print("Uploading sequences...")
    try:
        token = login()
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
        return
    session = requests.Session()
    session.headers.update({"Authorization": 'Bearer '+token})
    sequences = pd.read_csv(SEQUENCE_RECORDS)
    try:
        jobs_res = session.get(SERVER_URL+'/api/job')
        jobs_res.raise_for_status()
        jobs = jobs_res.json()
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
        return
    counter = MAX_UPLOADS - len(jobs)
    if counter < 0:
        counter = 0
    print('Will upload a max of '+str(counter)+' sequences')
    for ix, s in sequences.iterrows():
        if counter == 0:
            break
        if s['sent']:
            continue
        sequences['sent'][ix] = upload_sequence(s, session)
        counter -= sequences['sent'][ix]
        time.sleep(INTERVAL_OPS)
    print('Uploading done')
    sequences.to_csv(SEQUENCE_RECORDS, index = False)

    dt = datetime.datetime.now() + datetime.timedelta(seconds = INTERVAL_LD)
    dt_string = dt.strftime("%d/%m/%Y %H:%M:%S")
    print('Next run at '+dt_string)
    
    scheduler.enter(INTERVAL_LD, 1, upload_new_sequences, (scheduler,))
    
def download_labelled_sequences(scheduler):
    dt_string = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print('Running at '+dt_string)

    print('Downloading sequences...')
    try:
        token = login()
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
        return
    session = requests.Session()
    session.headers.update({"Authorization": 'Bearer '+token})
    sequences = pd.read_csv(SEQUENCE_RECORDS)
    try:
        jobs_res = session.get(SERVER_URL+'/api/job')
        jobs_res.raise_for_status()
        jobs = jobs_res.json()
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
        return
    complete_jobs = [j for j in jobs if j['completion'] == 1]
    print('Will download '+str(len(complete_jobs))+' sequences')
    for job in complete_jobs:
        print('Downloading sequence '+job['dataset']['name'])
        dataset_name = job['dataset']['name']
        dataset_id = job['dataset']['_id']
        job_id = job['_id']
        try:
            labels_res = session.get(SERVER_URL+'/labels/'+dataset_id+'/labels/'+job_id+'_labels.csv')
            labels_res.raise_for_status()
        except HTTPError as http_err:
            print(f'HTTP error occurred: {http_err}')
            continue
        open(SEQUENCE_LABELS+'/'+dataset_name+'.csv', 'wb').write(labels_res.content)
        sequences.loc[sequences['name'] == dataset_name, 'labeled'] = True
        try:
            dataset_delete_res = session.delete(SERVER_URL+'/api/dataset/'+dataset_id)
            dataset_delete_res.raise_for_status()
        except HTTPError as http_err:
            print(f'HTTP error occurred: {http_err}')
            continue
        time.sleep(INTERVAL_OPS)
    sequences.to_csv(SEQUENCE_RECORDS, index = False)
    print('Downloading done')
    
    scheduler.enter(INTERVAL_LD, 1, download_labelled_sequences, (scheduler,))

def main():
    s = sched.scheduler(time.time, time.sleep)
    s.enter(1, 1, download_labelled_sequences, (s,))
    s.enter(1, 1, upload_new_sequences, (s,))
    s.run()

if __name__ == "__main__":
    main()