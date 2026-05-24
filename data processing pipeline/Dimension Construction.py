import pandas as pd
import requests
import json
import os

def get_qwen_completion(prompt):
    API_URL = "https://api.openai.com/v1/chat/completions"
    API_KEY = "xxxx"
    MODEL_NAME = "gpt-4o"
    HEADERS = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=300)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        raise RuntimeError(f"GPT API 返回状态码: {response.status_code}, 响应内容: {response.text}")

# List_of_Target.csv: 需要生成 dimension 的 target 列表
target_list_df = pd.read_csv('List_of_Target.csv')
target = target_list_df['Target'].iloc[0]

# xxx_data.csv: 与 target 对应的原始数据集（含 Tweet 与 Stance）
tweet_df = pd.read_csv('xxx_data.csv')
tweet_corpus = tweet_df.apply(lambda row: f"Target: {row['Target']} | Tweet: {row['Tweet']} | Stance: {row['Stance']}", axis=1).tolist()

# 简单敏感词过滤，只保留不含敏感词的推文
sensitive_words = ['abortion', 'kill', 'hate', 'violence']
def is_safe_tweet(tweet):
    return not any(word in tweet.lower() for word in sensitive_words)
filtered_tweet_corpus = [t for t in tweet_corpus if is_safe_tweet(t)]

tweet_corpus_str = '\n'.join(filtered_tweet_corpus[:300])

for run_idx in range(1, 11):
    print(f"\n===== 第{run_idx}次运行 =====\n")
    # Step 1
    prompt_step1 = f"""
You are an expert in sociopolitical knowledge classification.

For subject classification, please refer to the following list of semantic domains:

[Religion, Philosophy, Ethics, Political Ideology, Science, Law, Gender Studies, Cultural Belief Systems, Human Rights, Technology, Environmentalism, Activism, Social Justice, Economic Theory, Psychological Orientation, Education, Health and Medicine, Conspiracy Theory, Spirituality, Popular Culture, Media Discourse, Other (please specify if necessary)]

Now, based on your common sense and public discourse, determine which of the following domains the following target belongs to:

Target: {target}

The following is the content that needs to be output in format.
Output in strict JSON format:

{{
"Target": "{target}",
"Domains": [<best matching domain list>]
}}
"""
    step1_output = get_qwen_completion(prompt_step1)
    print("\n===== Step 1 Output =====\n")
    print(step1_output)

    # Step 2
    try:
        step1_json = json.loads(step1_output)
        step1_target = step1_json.get("Target", target)
        step1_domains = step1_json.get("Domains", [])
    except Exception as e:
        step1_target = target
        step1_domains = []
    prompt_step2 = f"""
You are a research assistant building a target-specific user profiling schema for stance detection.

TASK OVERVIEW:

Your task is to identify what user attributes (i.e., profile dimensions or stance expression cues) are most relevant for understanding how users take a stance toward a given ideological or controversial topic — called the **Target**.

INPUTS:

1. Target: "{step1_target}"

2. Target Domains: This target belongs to the following high-level semantic domains based on structured knowledge classification:
{step1_domains}
(e.g., Religion, Ethics, Philosophy, etc.)

3. Tweet Corpus:
Below is a large sample of real social media posts (tweets) discussing the Target. Each line contains Target, Tweet, and Stance information. Please use this corpus to discover common cues or behaviors people display when expressing a stance.

{tweet_corpus_str}

---

INSTRUCTIONS:

Please analyze this data and generate a structured list of user profiling dimensions that should be tracked or inferred in order to understand their stance toward the given Target.

Each dimension should represent a field of user behavior, belief, language use, or interaction pattern that you infer to be useful based on:
- The target's domain-specific characteristics
- Common discourse patterns in the tweets

OUTPUT FORMAT (strict JSON):

{{
  "Target": "<TARGET_NAME>",
  "ProfilingDimensions": [
    {{
      "DimensionName": "Religious Identity",
      "Rationale": "Religious identity is a common way to express a stance toward the target."...,
      "TweetExamples": [
        "I left religion years ago and never looked back.",
        "Being an atheist doesn't make you immoral."...,
      ]
    }}
  ]
}}
"""
    step2_output = get_qwen_completion(prompt_step2)
    print("\n===== Step 2 Output =====\n")
    print(step2_output)

    # Step 3
    prompt_step3 = f"""
You are now acting as a stance profiling analyst. Your task is to evaluate a list of user profiling dimensions for their utility in identifying user stance toward a given ideological target.

---

TASK OVERVIEW:

You are provided with:
- A Target: "{step1_target}"
- A list of user profiling dimensions generated from real user data and discourse (Tweets).
Each profiling dimension represents a possible attribute that could help infer a user's stance toward this target.

---

YOUR GOAL:

For each profiling dimension, you must evaluate it across three key criteria:

1. Observability: Is this attribute easily detectable or inferable from a user's public social media data (e.g., their tweets, hashtags, follows, interactions)?
   - [High / Medium / Low]

2. Discriminativeness: Does this attribute vary meaningfully between users with different stances on the target (e.g., FAVOR vs. AGAINST)?
   - [High / Medium / Low]

3. Generalizability: Could this attribute be useful for other, related topics (e.g., religious discourse like "Islam" or "Secularism")?
   - [High / Medium / Low]

---

Please strictly follow the output format requirements:
```json
{{
  "ProfilingDimensions": [
    {{
      "DimensionName": "xxx",
      "Observability": "High / Medium / Low",
      "Discriminativeness": "High / Medium / Low",
      "Generalizability": "High / Medium / Low" 
    }}
  ]
}}  
```

INPUTS:

Target: {step1_target}

Profiling Dimensions:
```json
{step2_output}
```
"""
    step3_output = get_qwen_completion(prompt_step3)
    print("\n===== Step 3 Output =====\n")
    print(step3_output)

    # 选出所有维度按High数量降序排序，取前5个，保存为high_dimensions.json
    ranked_dimensions = []
    try:
        import re
        match = re.search(r'```json(.*?)```', step3_output, re.DOTALL)
        json_str = match.group(1).strip() if match else step3_output
        parsed = json.loads(json_str)
        dims = parsed.get('ProfilingDimensions', []) if isinstance(parsed, dict) else parsed
        for dim in dims:
            obs = dim.get('Observability') or dim.get('Observability (High / Medium / Low)')
            dis = dim.get('Discriminativeness') or dim.get('Discriminativeness (High / Medium / Low)')
            gen = dim.get('Generalizability') or dim.get('Generalizability (High / Medium / Low)')
            high_count = sum([
                1 if obs and obs.strip().lower() == 'high' else 0,
                1 if dis and dis.strip().lower() == 'high' else 0,
                1 if gen and gen.strip().lower() == 'high' else 0
            ])
            dim['HighCount'] = high_count
            ranked_dimensions.append(dim)
        # 按HighCount降序，取前5
        ranked_dimensions = sorted(ranked_dimensions, key=lambda d: d['HighCount'], reverse=True)[:5]
        high_json = {"Target": step1_target, "HighDimensions": ranked_dimensions}
        with open('high_dimensions.json', 'w', encoding='utf-8') as f:
            json.dump(high_json, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"解析或保存high维度失败: {e}\n原始输出: {step3_output}")

    # Step 4
    try:
        with open('high_dimensions.json', 'r', encoding='utf-8') as f:
            high_json = json.load(f)
            high_dims = high_json.get('HighDimensions', [])
    except Exception as e:
        high_dims = []
    prompt_step4 = f"""
You are now performing data source analysis for a user profiling system.

---

TASK CONTEXT:

You have already completed stance-related profiling for a controversial target:
- Target: {step1_target}
- You have a list of high-value profiling dimensions relevant for identifying user stance.

Each dimension describes a stance-informative user attribute.

---

YOUR GOAL:

For each profiling dimension, assess:
Required Supplementary Sources: If not fully observable from text, what data sources would be necessary to infer or complete it?
   Examples: 
   - Retweet behavior
   - Follower/following list
   - Hashtag use history
   - User bio or profile
   - Temporal tweet history
   - External profile links
   - etc.

---

## OUTPUT FORMAT:

Return a table in JSON format like this:

```json
[
  {{
    "DimensionName": "Religious Identity or Affiliation",
    "SupplementarySources": ["User bio", "Historical tweet patterns"],
  }},
  {{
    "DimensionName": "Community Affiliation via Hashtags",
    "SupplementarySources": ["Hashtag usage history"],
  }},
  {{
    "DimensionName": "Retweet Behavior of Atheist Influencers",
    "SupplementarySources": ["Retweet logs", "Interaction metadata"],
  }}
]
```

High-value Profiling Dimensions:
{json.dumps(high_dims, ensure_ascii=False, indent=2)}
"""
    step4_output = get_qwen_completion(prompt_step4)
    print("\n===== Step 4 Output (Final) =====\n")
    print(step4_output)

    # 保存为格式化json文件
    try:
        import re
        match = re.search(r'```json(.*?)```', step4_output, re.DOTALL)
        json_str = match.group(1).strip() if match else step4_output
        parsed = json.loads(json_str)
        safe_target = step1_target.replace(' ', '_').replace('/', '_')
        outfilename = f"{safe_target}_dimension{run_idx}.json"
        with open(outfilename, 'w', encoding='utf-8') as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        print(f"\n已保存最终输出为: {outfilename}")
    except Exception as e:
        print(f"\n保存最终输出为json文件失败: {e}")
