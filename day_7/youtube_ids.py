# If you want to load all YouTube videos from a specific channel in one go, use these functions.

import googleapiclient.discovery
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

api_key = "Your Google Dev API Key" #@param {type:"string"}
channel_id = "" #@param {type:"string"} # Get your channel ID here https://commentpicker.com/youtube-channel-id.php

def get_channel_videos(channel_id, api_key):
    youtube = googleapiclient.discovery.build(
        "youtube", "v3", developerKey=api_key)

    video_ids = []
    page_token = None

    while True:
        request = youtube.search().list(
            part="snippet",
            channelId=channel_id,
            maxResults=10,  # Fetch 50 videos at a time
            pageToken=page_token  # Add pagination
        )
        response = request.execute()

        video_ids += [item['id']['videoId'] for item in response['items'] if item['id']['kind'] == 'youtube#video']
        
        # Check if there are more videos to fetch
        if 'nextPageToken' in response:
            page_token = response['nextPageToken']
        else:
            break

    return video_ids

def get_transcript(video_id):
    # Get transcript list
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    transcripts_manual = transcript_list._manually_created_transcripts

    # Get transcript. If no manually created transcript is available, use the automatically generated one.
    if transcripts_manual:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    else:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['nl', 'en'])

    # Format transcript as text
    formatter = TextFormatter()
    text_transcript = formatter.format_transcript(transcript)
    text_transcript = text_transcript.replace('\n', ' ')

    return text_transcript    

video_ids = get_channel_videos(channel_id, api_key)

transcript = get_transcript(video_ids[0])