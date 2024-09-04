import urllib.request
import os
from tqdm import tqdm
import json 
from multiprocessing import Pool


def load_vaani_files(json_file): 
    print('opening json file')
    with open(json_file) as file:
        file_contents = file.read()

    file_contents = file_contents[:8398429148] + ']'

    print(f'length of file contents: {len(file_contents)}')

    data = json.loads(file_contents)

    print('data loaded')

    hindi_data = []
    hindi_urls = []
    other = []


    print("finding hindi-only utterances")
    for i in tqdm(data):
        if i['metadata']['assertLanguage'] == 'HINDI':
            hindi_data.append(i)
            hindi_urls.append(i['file_url'])
        else:
            other.append(i)

    print(f'number of hindi utterances: {len(hindi_data)}')
    return hindi_data, hindi_urls


def load_urls(urls):
    data_path = '/fs/clip-scratch/jrodri20/corpora/data/'
    for link in tqdm(urls):
        filename = link.split('/')[-1]
        output_file = os.path.join(data_path, filename)
        if not os.path.exists(output_file):
            urllib.request.urlretrieve(link, output_file)

if __name__ == '__main__':
    hindi_data, hindi_urls = load_vaani_files('/fs/nexus-scratch/jrodri20/corpora/hindi/Full_vaani_data.json')
    n = 10
    chunk_size = int(len(hindi_urls) / 10)

    hindi_url_chunks = []
    for i in range(1, 11):
        print(f'processing indices: {chunk_size*(i-1)} to {chunk_size*i}')
        if i == 1:
            hindi_url_chunks.append(hindi_urls[:chunk_size])
        else:
            hindi_url_chunks.append(hindi_urls[chunk_size*(i-1):chunk_size*i])
        
    with Pool(n) as p:
        print(p.map(load_urls, hindi_url_chunks))


