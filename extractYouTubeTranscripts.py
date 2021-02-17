import numpy as np
import pandas as pd

import urllib.request
import json
import time
import argparse

from youtube_transcript_api import YouTubeTranscriptApi

# Note: An API key is only necessary when getting videos from a channel
global api_key
api_key = ''


def get_channel_videos(channel_id, delay=0):

    if api_key == '':
        raise Exception("Must specify an API key. See README.md for more info.")
    
    base_search_url = 'https://www.googleapis.com/youtube/v3/search?'
    first_url = base_search_url+'key={}&channelId={}&part=snippet,id&order=date&maxResults=25'.format(api_key, channel_id)
    titles = []
    video_links = []
    url = first_url
    while True:
        time.sleep(delay)
        inp = urllib.request.urlopen(url)
        resp = json.load(inp)
        for i in resp['items']:
            if i['id']['kind'] == "youtube#video":
                title = i['snippet']['title']
                if 'Season' in title and 'Episode' in title:
                    titles.append(title)
                    video_links.append(i['id']['videoId'])

        try:
            next_page_token = resp['nextPageToken']
            url = first_url + '&pageToken={}'.format(next_page_token)
        except:
            break
    return titles, video_links


def clean_links(links):
    if isinstance(links, list):
        links = np.array(links)

    clean_link = lambda l: l[l.index("v=") + 2:] if 'youtube' in l.lower() else l
    f = np.vectorize(clean_link)
    return f(links)


def load_transcript(link):
    try:
        d = YouTubeTranscriptApi.get_transcript(link)
        return ' '.join([i['text'] for i in d]), True
    except:
        return '', False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='Path to save all tanscript .txt data')
    parser.add_argument('-l', '--links', help='Path for csv file containing video titles and links', default=None)
    parser.add_argument('-c', '--channel', help='Download all video transcripts from a given channel ID (Requires API key)', default=None)
    parser.add_argument('-s', '--separate', help="Path to save transcripts as separate .txt files", default="transcripts/")

    args = parser.parse_args()

    # Process arguments
    args.data = args.data + '.txt' if args.data[-4:] != '.txt' else args.data
    args.separate = args.separate + '/' if args.separate[-1] != '/' else args.separate

    # Get all YouTube video links from a given YouTube channel
    if args.channel is not None:
        titles, links = get_channel_videos(args.channel, delay=0.05)
        d = {'title': titles, 'link': links} 
        df = pd.DataFrame(d)
        df.to_csv('links.csv', index=False)

    # Get YouTube video links from a csv file
    elif args.links is not None:
        df = pd.read_csv(args.links)
        titles = df.title.values
        links = df.link.values
    else:
        raise Exception("Use either '--links' or '--channel' to specify YouTube videos to load. See README.md for more info.")

    links = clean_links(links)

    # Load transcripts
    data_file = open(args.data, "w", encoding="utf-8")
    for count, link in enumerate(links):
        # Get transcript
        transcript, passed = load_transcript(link)

        if passed:
            # Save transcript to separate file
            text_file = open(args.separate + titles[count] + ".txt", "w", encoding="utf-8")
            text_file.write(transcript)
            text_file.close()

            # Save transcript to main file
            data_file.write(transcript + '\n')
            print("Done processing ({}/{}):".format(count+1, len(links)), titles[count])
        
        else:
            print("Couldn't get transcript ({}/{}):".format(count+1, len(links)), titles[count])

    data_file.close()


if __name__ == "__main__":
    main()
