import pandas as pd
import requests
import json
import re

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


def parse_json_output(text):
    match = re.search(r'```json(.*?)```', text, re.DOTALL)
    json_str = match.group(1).strip() if match else text
    return json.loads(json_str)


def build_aspect_template(dimensions):
    aspect_lines = []
    for dim in dimensions:
        if isinstance(dim, dict):
            name = dim.get("DimensionName", "")
        else:
            name = str(dim)
        if name:
            aspect_lines.append(f'  {{"DimensionName": "{name}", "Value": ""}}')
    return "[\n" + ",\n".join(aspect_lines) + "\n]"


# xxx_data.csv: 原始数据集，含 Target、Tweet、Stance，用于逐条构建 user profile
tweet_df = pd.read_csv('xxx_data.csv')

# xxx_dimension.json: Dimension Construction 步骤产出的 target-specific dimension 列表（top-10）
with open('xxx_dimension.json', 'r', encoding='utf-8') as f:
    target_dimensions = json.load(f)

target = tweet_df['Target'].iloc[0]
safe_target = target.replace(' ', '_').replace('/', '_')
aspect_template = build_aspect_template(target_dimensions)

target_definition_cache = {}
socio_cultural_context_cache = {}
profile_results = []

for idx, row in tweet_df.iterrows():
    tweet = row['Tweet']
    target = row['Target']
    stance = row['Stance']

    print(f"\n===== 处理第 {idx + 1}/{len(tweet_df)} 条 =====")
    print(f"Target: {target} | Stance: {stance}")
    print(f"Tweet: {tweet[:100]}...")

    # Step 1: Target Definition
    if target in target_definition_cache:
        step1_output = target_definition_cache[target]
        print("\n[Step 1] 从缓存获取 Target Definition")
    else:
        prompt_step1 = f"""Target of Interest: {target}

Provide a structured explanation that includes:
- A clear and concise definition
- The disciplinary domains it belongs to (e.g., politics, religion, philosophy)
- Its significance in contemporary public or academic debates

Output at least 80 words.
"""
        step1_output = get_qwen_completion(prompt_step1)
        step1_output = f"{target}: {step1_output}"
        target_definition_cache[target] = step1_output
    print("\n===== Step 1 Output (Target Definition) =====\n")
    print(step1_output)

    # Step 2: User Background Analysis (Socio-Cultural Context)
    step2_cache_key = (tweet, step1_output)
    if step2_cache_key in socio_cultural_context_cache:
        step2_output = socio_cultural_context_cache[step2_cache_key]
        print("\n[Step 2] 从缓存获取 Socio-Cultural Context")
    else:
        prompt_step2 = f"""Here is a structured definition of the target:
{step1_output}

Now, consider the following tweet as a representative example of online discourse:

Tweet: {tweet}

Please conduct a comprehensive socio-cultural context analysis based on the structured definition of the target, including but not limited to:
- How is this concept perceived and debated in different cultural or regional contexts?
- What ideological, historical, or political tensions surround it?
- What kinds of individuals or groups typically engage with it?
- In what ways might online discourse reflect or distort public opinion on this target?

You may briefly mention the tweet as a lens or micro-example, but please keep the focus primarily on the target.

Output at least 100 words.
"""
        step2_output = get_qwen_completion(prompt_step2)
        socio_cultural_context_cache[step2_cache_key] = step2_output
    print("\n===== Step 2 Output (User Background Analysis) =====\n")
    print(step2_output)

    # Step 3: Motivation Analysis
    prompt_step3 = f"""Here is a socio-cultural context of the target:
{step2_output}

The tweet and the stance the user takes are as follows:

Tweet: {tweet}
Stance: {stance}

Using the above context, infer:
- What belief systems or ideological positions could explain why a user holds this stance toward the target.
- How the user's interpretation of the target may interact with current events, sociocultural dynamics, or personal identity.
- Whether the stance is likely driven by reaction, affiliation, conviction or other motivations.

Make sure your explanation connects the **stance to the socio-cultural context of the target**, not just to the tweet wording.

Output at least 100 words.
"""
    step3_output = get_qwen_completion(prompt_step3)
    print("\n===== Step 3 Output (Motivation Analysis) =====\n")
    print(step3_output)

    # Step 4: Rhetorical Analysis
    prompt_step4 = f"""Here is a socio-cultural context of the target:
{step2_output}

The tweet and the stance the user takes are as follows:

Tweet: {tweet}
Stance: {stance}

Please analyze the rhetorical and emotional features of the following tweet:
- Indicate whether the language is emotionally charged, neutral, sarcastic, authoritative, uncertain, etc.
- Explain what lexical or stylistic choices support this assessment.
- Relate the rhetorical tone to the user's stance: does it reinforce their position or obscure it?

Output at least 50 words.
"""
    step4_output = get_qwen_completion(prompt_step4)
    print("\n===== Step 4 Output (Rhetorical Analysis) =====\n")
    print(step4_output)

    # Step 5: Profile Generation — Natural Language Description
    prompt_step5 = f"""Please write a concise, coherent and high-information user background description based on the following structured information:
- Focus on describing a user who holds a "{stance}" stance on the target topic {target}, and his possible values, social background or identity tendencies;
- Tweet is the context for expressing his views. Please make a comprehensive judgment based on its tone, style and cultural context;
- The analysis should reflect why the user holds this stance, and cannot just repeat the surface content of the tweet;
- The output should be natural and true, and conform to the possible thinking characteristics of real social media users.

Structured reference information:

1. Target definition and category:
{step1_output}
2. Socio-Cultural Context:
{step2_output}

3. Stance Motivation:
{step3_output}

4. Rhetorical and Emotional Tone:
{step4_output}
"""
    step5_output = get_qwen_completion(prompt_step5)
    print("\n===== Step 5 Output (Natural Language Description) =====\n")
    print(step5_output)

    # Step 5 (cont.): Profile Generation — Structured JSON mapped to dimensions
    prompt_step6 = f"""User background description:
{step5_output}

Based only on the following user background description, please output a structured JSON object that includes the following dimensions:
- For each dimension, if the content can be directly obtained from the description, fill it directly.
- If it cannot be directly obtained, please make a reasonable inference based on the description and fill it in. Ensure that every dimension is completed and nothing is left blank.
- Avoid vague or generic statements such as "no evidence," "not mentioned," or "indirect."
- The description of each dimension should be as close as possible to the situation of social platforms in the real world.
- IMPORTANT: The final output must NOT contain any explicit mention of the stance field values (NONE, FAVOR, AGAINST, NEUTRAL, etc.) in any form.

The JSON dimensions are as follows:

Basic dimensions:
[
  {{"DimensionName": "Display Name", "Value": ""}},
  {{"DimensionName": "Age", "Value": ""}},
  {{"DimensionName": "User Bio", "Value": ""}},
  {{"DimensionName": "Location", "Value": ""}},
  {{"DimensionName": "Website", "Value": ""}},
  {{"DimensionName": "Join Date", "Value": ""}},
  {{"DimensionName": "Following and Followers", "Value": ""}}
]

Target-specific dimensions:
{aspect_template}
"""
    step6_output = get_qwen_completion(prompt_step6)
    print("\n===== Step 5 Output (Structured User Profile) =====\n")
    print(step6_output)

    try:
        user_profile = parse_json_output(step6_output)
    except Exception as e:
        print(f"\n解析结构化 user profile 失败: {e}")
        user_profile = step6_output

    record = {
        "Target": target,
        "Tweet": tweet,
        "Stance": stance,
        "Natural Language Description": step5_output,
        "User Profile": user_profile
    }
    profile_results.append(record)

    outfilename = f"{safe_target}_profile_{idx + 1}.json"
    try:
        with open(outfilename, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print(f"\n已保存单条 user profile 为: {outfilename}")
    except Exception as e:
        print(f"\n保存单条 user profile 失败: {e}")

# 保存汇总结果
try:
    profiles_csv = f"{safe_target}_profiles.csv"
    profiles_json = f"{safe_target}_profiles.json"

    summary_df = pd.DataFrame([
        {
            "Target": r["Target"],
            "Tweet": r["Tweet"],
            "Stance": r["Stance"],
            "Natural Language Description": r["Natural Language Description"],
            "User_Profile": json.dumps(r["User Profile"], ensure_ascii=False)
        }
        for r in profile_results
    ])
    summary_df.to_csv(profiles_csv, index=False)

    with open(profiles_json, 'w', encoding='utf-8') as f:
        json.dump(profile_results, f, ensure_ascii=False, indent=2)

    print(f"\n已保存汇总 CSV 为: {profiles_csv}")
    print(f"已保存汇总 JSON 为: {profiles_json}")
except Exception as e:
    print(f"\n保存汇总结果失败: {e}")
