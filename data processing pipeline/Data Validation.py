import pandas as pd
import requests
import json
import re

def get_qwen_completion(prompt, max_tokens=10000):
    API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    API_KEY = "xxxx"
    MODEL_NAME = "qwen3-next-80b-a3b-instruct"
    HEADERS = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": max_tokens
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=300)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        raise RuntimeError(f"Qwen API 返回状态码: {response.status_code}, 响应内容: {response.text}")


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


def get_other_stances(current_stance):
    all_stances = ['FAVOR', 'NONE', 'AGAINST']
    current = current_stance.upper()
    others = [s for s in all_stances if s != current]
    return others[0], others[1]


def parse_final_decision(text):
    decision_match = re.search(r'Final (?:Decision|Stance|Label):\s*\*?\*?(FAVOR|NONE|AGAINST)\*?\*?', text, re.IGNORECASE)
    if decision_match:
        return decision_match.group(1).upper()
    decision_match = re.search(r'\*\*Final (?:Decision|Stance|Label):\s*(FAVOR|NONE|AGAINST)\*\*', text, re.IGNORECASE)
    if decision_match:
        return decision_match.group(1).upper()
    return None


def parse_ranking_output(text, valid_titles):
    raw_items = []

    ranking_match = re.search(r'(?:Validated Ranking|Final Ranking|Ranking):\s*\{([^}]+)\}', text, re.IGNORECASE)
    if ranking_match:
        raw_items = [item.strip().strip('"').strip("'") for item in ranking_match.group(1).split(',')]
    else:
        ranking_match = re.search(r'(?:Validated Ranking|Final Ranking|Ranking):\s*([^\n]+)', text, re.IGNORECASE)
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


def validate_sample(tweet, target, user_profile, stance, dimension_ranking):
    stance2, stance3 = get_other_stances(stance)
    dimension_titles = extract_target_specific_titles(user_profile)
    dimension_titles_str = ", ".join([f'"{title}"' for title in dimension_titles])

    # Step 1: Supportive Reasoning
    prompt_step1 = f"""You are an AI expert in judgement for stance detection.

INPUTS:
1. Target: {target}
2. Tweet: {tweet}
3. User Profile: {user_profile}
4. Stance: {stance}
5. Dimension Ranking: {dimension_ranking}

YOUR GOAL:
Assume the given stance is correct. Explain why this stance can be justified based on the tweet and user profile. In your explanation, provide step-by-step reasoning considering:
1. Profile cues: which dimensions of the user profile support this stance (e.g., beliefs, orientation, past engagement).
2. Tweet cues: which words, phrases, or implications in the tweet support the stance?
3. Consistency: how the combination of tweet and profile aligns logically with the stance.

Provide clear and evidence-based reasons without inventing facts. Output 2-4 concise numbered supportive points.

Output format:
Supportive Reasons:
1. ...
2. ...
"""
    supportive_reasons = get_qwen_completion(prompt_step1, max_tokens=2000)
    print("\n===== Step 1 Output (Supportive Reasoning) =====\n")
    print(supportive_reasons)

    # Step 2: Critical Reasoning (incorrect stance 1)
    prompt_step2 = f"""You are an AI expert in judgement for stance detection.

INPUTS:
1. Target: {target}
2. Tweet: {tweet}
3. User Profile: {user_profile}
4. Stance: {stance2}
5. Dimension Ranking: {dimension_ranking}

YOUR GOAL:
Assume the given stance is incorrect. Critically analyze why this stance may not be justified based on the tweet and user profile. In your analysis, provide step-by-step reasoning considering:
1. Contradictory profile cues: which dimensions of the profile may oppose this stance.
2. Contradictory tweet cues: words, phrases, or implications that weaken this stance.
3. Unsupported assumptions: any leaps or reasoning in the annotation not grounded in the inputs.

Provide 2-4 numbered oppositive points, concise and evidence-based. Do not invent facts.

Output format:
Oppositive Reasons:
1. ...
2. ...
"""
    oppositive_reasons_1 = get_qwen_completion(prompt_step2, max_tokens=2000)
    print(f"\n===== Step 2 Output (Critical Reasoning: {stance2}) =====\n")
    print(oppositive_reasons_1)

    # Step 3: Critical Reasoning (incorrect stance 2)
    prompt_step3 = f"""You are an AI expert in judgement for stance detection.

INPUTS:
1. Target: {target}
2. Tweet: {tweet}
3. User Profile: {user_profile}
4. Stance: {stance3}
5. Dimension Ranking: {dimension_ranking}

YOUR GOAL:
Assume the given stance is incorrect. Critically analyze why this stance may not be justified based on the tweet and user profile. In your analysis, provide step-by-step reasoning considering:
1. Contradictory profile cues: which dimensions of the profile may oppose this stance.
2. Contradictory tweet cues: words, phrases, or implications that weaken this stance.
3. Unsupported assumptions: any leaps or reasoning in the annotation not grounded in the inputs.

Provide 2-4 numbered oppositive points, concise and evidence-based. Do not invent facts.

Output format:
Oppositive Reasons:
1. ...
2. ...
"""
    oppositive_reasons_2 = get_qwen_completion(prompt_step3, max_tokens=2000)
    print(f"\n===== Step 3 Output (Critical Reasoning: {stance3}) =====\n")
    print(oppositive_reasons_2)

    # Step 4: Reflective Judgement
    prompt_step4 = f"""You are an AI expert in judgement for stance detection.

INPUTS:
1. Target: {target}
2. Tweet: {tweet}
3. User Profile: {user_profile}
4. Original Stance Annotation: {stance}
5. Original Dimension Ranking: {dimension_ranking}

Supportive Reasons for {stance}:
{supportive_reasons}

Oppositive Reasons for {stance2}:
{oppositive_reasons_1}

Oppositive Reasons for {stance3}:
{oppositive_reasons_2}

YOUR GOAL:
Reflect on all three stances and the dimension ranking by evaluating their respective reasons. Your task is to determine the final label and dimension ranking for this sample. In your analysis, provide step-by-step reasoning considering:
1. Evaluate the supportive reasons: assess the validity and strength of the reasons supporting {stance} based on the tweet and user profile.
2. Evaluate the oppositive reasons for {stance2}: assess whether the reasons opposing {stance2} are valid.
3. Evaluate the oppositive reasons for {stance3}: assess whether the reasons opposing {stance3} are valid.
4. Make a final judgment: based on all evidence from the tweet and user profile, determine which label (FAVOR, NONE, or AGAINST) is most appropriate for this sample.
5. Analyze whether the dimension ranking is reasonable. If it is not reasonable, provide the corrected ranking.

Constraints:
- Only use information from the Tweet and User Profile.
- You MUST ONLY select ranking dimensions from: {dimension_titles_str}
- Output exactly 6 dimension titles, ranked from most to least influential.
- The final label MUST be one of: FAVOR, NONE, or AGAINST.

Output format:
Final Decision: {{FAVOR or NONE or AGAINST}}
Validated Ranking: {{dimension 1, dimension 2, dimension 3, dimension 4, dimension 5, dimension 6}}
"""
    reflective_output = get_qwen_completion(prompt_step4, max_tokens=10000)
    print("\n===== Step 4 Output (Reflective Judgement) =====\n")
    print(reflective_output)

    final_decision = parse_final_decision(reflective_output) or stance.upper()
    validated_ranking = parse_ranking_output(reflective_output, dimension_titles)

    if not validated_ranking and dimension_ranking:
        validated_ranking = parse_ranking_output(dimension_ranking, dimension_titles)

    return {
        "Supportive_Reasons": supportive_reasons,
        "Oppositive_Reasons_1": oppositive_reasons_1,
        "Oppositive_Reasons_2": oppositive_reasons_2,
        "Reflective_Output": reflective_output,
        "Final_Stance": final_decision,
        "Validated_Ranking": validated_ranking
    }


# xxx_ranking.csv: Explainability Assessment 步骤产出的汇总结果（含 Target、Tweet、Stance、User_Profile、Dimension_Ranking）
ranking_df = pd.read_csv('xxx_ranking.csv')

target = ranking_df['Target'].iloc[0]
safe_target = target.replace(' ', '_').replace('/', '_')
validated_results = []

for idx, row in ranking_df.iterrows():
    target = row['Target']
    tweet = row['Tweet']
    stance = row['Stance']
    user_profile = row['User_Profile']
    dimension_ranking = row.get('Dimension_Ranking', '')

    print(f"\n===== 验证第 {idx + 1}/{len(ranking_df)} 条 =====")
    print(f"Target: {target} | Stance: {stance}")

    try:
        result = validate_sample(tweet, target, user_profile, stance, dimension_ranking)
    except Exception as e:
        print(f"\n验证失败: {e}")
        result = {
            "Supportive_Reasons": "",
            "Oppositive_Reasons_1": "",
            "Oppositive_Reasons_2": "",
            "Reflective_Output": "",
            "Final_Stance": stance.upper() if isinstance(stance, str) else "",
            "Validated_Ranking": []
        }

    formatted_ranking = format_ranking(result["Validated_Ranking"]) if result["Validated_Ranking"] else dimension_ranking

    record = {
        "Target": target,
        "Tweet": tweet,
        "User_Profile": user_profile,
        "Stance": stance,
        "Dimension_Ranking": dimension_ranking,
        "Final_Stance": result["Final_Stance"],
        "Validated_Ranking": formatted_ranking
    }
    validated_results.append(record)

    outfilename = f"{safe_target}_validated_{idx + 1}.json"
    try:
        with open(outfilename, 'w', encoding='utf-8') as f:
            json.dump({**record, "Validation_Details": result}, f, ensure_ascii=False, indent=2)
        print(f"\n已保存单条 validation 结果为: {outfilename}")
    except Exception as e:
        print(f"\n保存单条 validation 结果失败: {e}")

try:
    validated_csv = f"{safe_target}_validated.csv"
    validated_json = f"{safe_target}_validated.json"

    summary_df = pd.DataFrame(validated_results)
    summary_df.to_csv(validated_csv, index=False)

    with open(validated_json, 'w', encoding='utf-8') as f:
        json.dump(validated_results, f, ensure_ascii=False, indent=2)

    print(f"\n已保存汇总 CSV 为: {validated_csv}")
    print(f"已保存汇总 JSON 为: {validated_json}")
except Exception as e:
    print(f"\n保存汇总结果失败: {e}")
