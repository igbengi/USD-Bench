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


def extract_target_specific_titles(user_profile):
    if isinstance(user_profile, str):
        profile_dict = json.loads(user_profile)
    else:
        profile_dict = user_profile

    for key in ["Target-specific dimensions", "Aspect information"]:
        items = profile_dict.get(key, [])
        if not items:
            continue
        titles = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if "DimensionName" in item:
                titles.append(item["DimensionName"])
            else:
                titles.extend(list(item.keys()))
        if titles:
            return titles
    return []


def parse_ranking_output(text, valid_titles):
    raw_items = []

    ranking_match = re.search(r'(?:Dimension Ranking|Ranking):\s*\{([^}]+)\}', text, re.IGNORECASE)
    if ranking_match:
        raw_items = [item.strip().strip('"').strip("'") for item in ranking_match.group(1).split(',')]
    else:
        ranking_match = re.search(r'(?:Dimension Ranking|Ranking):\s*([^\n]+)', text, re.IGNORECASE)
        if ranking_match:
            ranking_str = re.sub(r'\*\*', '', ranking_match.group(1).strip())
            raw_items = [item.strip().strip('"').strip("'") for item in ranking_str.split(',')]
        else:
            for line in text.splitlines():
                item_match = re.match(r'^\s*\d+\.\s*(.+)$', line.strip())
                if item_match:
                    raw_items.append(item_match.group(1).strip())

    valid_ranking = []
    for item in raw_items:
        if not item:
            continue
        cleaned = re.sub(r'\*\*', '', item).strip()
        for sep in [' – ', ' - ', ': ', ' — ']:
            if sep in cleaned:
                cleaned = cleaned.split(sep)[0].strip()
        if cleaned in valid_titles:
            valid_ranking.append(cleaned)
        else:
            for title in valid_titles:
                if cleaned.lower() == title.lower():
                    valid_ranking.append(title)
                    break

    return valid_ranking[:6]


def format_ranking(ranking):
    return "\n".join([f"{i + 1}. {dim}" for i, dim in enumerate(ranking)])


# xxx_profiles.csv: User Profile Construction 步骤产出的汇总结果（含 Target、Tweet、Stance、User_Profile）
profiles_df = pd.read_csv('xxx_profiles.csv')

target = profiles_df['Target'].iloc[0]
safe_target = target.replace(' ', '_').replace('/', '_')
ranking_results = []

for idx, row in profiles_df.iterrows():
    target = row['Target']
    tweet = row['Tweet']
    stance = row['Stance']
    user_profile = row['User_Profile']

    print(f"\n===== 处理第 {idx + 1}/{len(profiles_df)} 条 =====")
    print(f"Target: {target} | Stance: {stance}")

    dimension_titles = extract_target_specific_titles(user_profile)
    if not dimension_titles:
        print("警告: 无法从 user profile 中提取 target-specific dimension 标题，跳过该条")
        ranking_results.append({
            "Target": target,
            "Tweet": tweet,
            "User_Profile": user_profile,
            "Stance": stance,
            "Dimension_Ranking": ""
        })
        continue

    dimension_titles_str = ", ".join([f'"{title}"' for title in dimension_titles])

    prompt = f"""You are an AI expert in user profiles for stance detection.

INPUTS:
1. Target: {target}
2. Tweet: {tweet}
3. User Profile: {user_profile}
4. Stance: {stance}

YOUR GOAL:
Identify the top six dimensions from the user profile's "Target-specific dimensions" that most influenced your judgment. Rank them from most to least influential. Evidence must be verbatim or close paraphrase. Do not create new dimension names or use descriptions. Output exactly 6 dimension titles.

Constraints:
- You MUST ONLY select from the following dimension titles: {dimension_titles_str}
- You MUST use the EXACT title names as listed above (case-sensitive).
- Do NOT create new dimension names or use descriptions.

Output format:
Dimension Ranking: {{dimension 1, dimension 2, dimension 3, dimension 4, dimension 5, dimension 6}}
"""
    step_output = get_qwen_completion(prompt)
    print("\n===== Explainability Assessment Output =====\n")
    print(step_output)

    ranking = parse_ranking_output(step_output, dimension_titles)
    formatted_ranking = format_ranking(ranking) if ranking else ""

    ranking_results.append({
        "Target": target,
        "Tweet": tweet,
        "User_Profile": user_profile,
        "Stance": stance,
        "Dimension_Ranking": formatted_ranking
    })

    outfilename = f"{safe_target}_ranking_{idx + 1}.json"
    try:
        with open(outfilename, 'w', encoding='utf-8') as f:
            json.dump({
                "Target": target,
                "Tweet": tweet,
                "Stance": stance,
                "Dimension_Ranking": ranking,
                "Raw_Output": step_output
            }, f, ensure_ascii=False, indent=2)
        print(f"\n已保存单条 ranking 为: {outfilename}")
    except Exception as e:
        print(f"\n保存单条 ranking 失败: {e}")

try:
    ranking_csv = f"{safe_target}_ranking.csv"
    ranking_json = f"{safe_target}_ranking.json"

    summary_df = pd.DataFrame(ranking_results)
    summary_df.to_csv(ranking_csv, index=False)

    with open(ranking_json, 'w', encoding='utf-8') as f:
        json.dump(ranking_results, f, ensure_ascii=False, indent=2)

    print(f"\n已保存汇总 CSV 为: {ranking_csv}")
    print(f"已保存汇总 JSON 为: {ranking_json}")
except Exception as e:
    print(f"\n保存汇总结果失败: {e}")
