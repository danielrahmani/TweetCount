#datacollection.py

from utils import *

working_dir = os.getcwd()
#sys.argv = ['','vp']
handle = usr_id_dict[item['user']['id']]
#handle='realdonaldtrump' #takes target twitter screenname as command-line argument
txtfilepath = working_dir + '/'+handle+'.txt'
jsonfilepath = working_dir + '/'+handle+'.p'

# Twitter authentication
twitter=Twython(APP_KEY,APP_SECRET,oauth_version=2) #simple authentication object
ACCESS_TOKEN=twitter.obtain_access_token()
twitter=Twython(APP_KEY,access_token=ACCESS_TOKEN)
 
#adapted from http://www.craigaddyman.com/mining-all-tweets-with-python/
#user_timeline=twitter.get_user_timeline(screen_name=handle,count=200) #if doing 200 or less, just do this one line
user_timeline = twitter.get_user_timeline(screen_name=handle,count=1,tweet_mode='extended') #get most recent tweet
max_id = user_timeline[0]['id']+1 # id for the most recent posted tweet

user_timeline = []
with open(jsonfilepath,'r') as fid: tw = json.load(fid) # read json file for previous tweets

since_id = tw[-1]['id']         # id of the last saved tweet

incremental = []
loop_count = 0
while  not (loop_count and not len(incremental)>199):
    if loop_count:
        print('sleeping')
        time.sleep(75)
    incremental = twitter.get_user_timeline(screen_name=handle,count=200, include_retweets=True, max_id=max_id,since_id=since_id,tweet_mode='extended')
    if incremental:
        user_timeline.extend(incremental)
        max_id = user_timeline[-1]['id']-1
    loop_count += 1

user_timeline = reversed(user_timeline) # reverse the time line


# save the updated tweets
with open(txtfilepath,'a') as fid:      
      for entry in user_timeline:
            t = tweet_obj(entry) 
            tw.append(t.__dict__)    
            fid.write(t.date)
            fid.write("\n")


# update the json file
with open(jsonfilepath,'w') as fid:
      json.dump(tw,fid)

