"""
Generate 100 English + 100 Chinese motion description pairs for human evaluation.

These are PAIRED: each Chinese prompt is the translation of the English one.
This lets you compare how models handle the same meaning in different languages.

Categories:
- Simple actions (walk, run, jump, sit, stand)
- Body part specific (wave hand, kick leg, turn head)
- Complex/compound actions (walk then sit, jump and wave)
- Direction/style (walk slowly, run in a circle, step backward)
"""

import json

# 100 English-Chinese paired prompts
# Organized by complexity: simple → complex
prompts = [
    # === SIMPLE LOCOMOTION (1-15) ===
    ("a person walks forward", "一个人向前走"),
    ("a person runs forward", "一个人向前跑"),
    ("a person walks slowly", "一个人慢慢走"),
    ("a person walks backward", "一个人向后走"),
    ("a person jogs in place", "一个人原地慢跑"),
    ("a person walks to the left", "一个人向左走"),
    ("a person walks to the right", "一个人向右走"),
    ("a person walks in a circle", "一个人绕圈走"),
    ("a person runs quickly", "一个人快速跑"),
    ("a person takes a step forward", "一个人向前迈一步"),
    ("a person walks with long strides", "一个人大步走"),
    ("a person tiptoes forward", "一个人踮脚向前走"),
    ("a person marches in place", "一个人原地踏步"),
    ("a person shuffles to the side", "一个人侧身移动"),
    ("a person paces back and forth", "一个人来回踱步"),

    # === JUMPING/VERTICAL (16-25) ===
    ("a person jumps up", "一个人跳起来"),
    ("a person hops on one foot", "一个人单脚跳"),
    ("a person jumps forward", "一个人向前跳"),
    ("a person does a small jump", "一个人小跳一下"),
    ("a person jumps and lands", "一个人跳起来然后落地"),
    ("a person crouches down then jumps", "一个人蹲下然后跳起来"),
    ("a person jumps to the right", "一个人向右跳"),
    ("a person jumps with both feet", "一个人双脚跳"),
    ("a person leaps forward", "一个人向前跃"),
    ("a person bounces up and down", "一个人上下弹跳"),

    # === SITTING/STANDING (26-35) ===
    ("a person sits down", "一个人坐下"),
    ("a person sits down on a chair", "一个人坐到椅子上"),
    ("a person stands up", "一个人站起来"),
    ("a person stands still", "一个人站着不动"),
    ("a person squats down", "一个人蹲下"),
    ("a person kneels down", "一个人跪下"),
    ("a person sits on the ground", "一个人坐在地上"),
    ("a person stands on one leg", "一个人单腿站立"),
    ("a person gets up from the ground", "一个人从地上站起来"),
    ("a person leans against a wall", "一个人靠在墙上"),

    # === ARM MOVEMENTS (36-50) ===
    ("a person waves with their right hand", "一个人用右手挥手"),
    ("a person waves with their left hand", "一个人用左手挥手"),
    ("a person raises both arms", "一个人举起双臂"),
    ("a person claps their hands", "一个人拍手"),
    ("a person stretches their arms", "一个人伸展手臂"),
    ("a person reaches up with one hand", "一个人单手向上伸"),
    ("a person puts their hands on their hips", "一个人双手叉腰"),
    ("a person crosses their arms", "一个人双臂交叉"),
    ("a person points forward", "一个人向前指"),
    ("a person raises their right hand", "一个人举起右手"),
    ("a person swings their arms", "一个人摆动手臂"),
    ("a person pushes something forward", "一个人向前推东西"),
    ("a person pulls something toward them", "一个人把东西拉向自己"),
    ("a person throws something", "一个人扔东西"),
    ("a person catches something", "一个人接住东西"),

    # === LEG MOVEMENTS (51-60) ===
    ("a person kicks with the right leg", "一个人用右腿踢"),
    ("a person kicks with the left leg", "一个人用左腿踢"),
    ("a person lifts their right leg", "一个人抬起右腿"),
    ("a person lifts their left leg", "一个人抬起左腿"),
    ("a person stomps their foot", "一个人跺脚"),
    ("a person steps over something", "一个人跨过东西"),
    ("a person lunges forward", "一个人向前弓步"),
    ("a person does a side kick", "一个人侧踢"),
    ("a person swings their leg", "一个人摆腿"),
    ("a person taps their foot", "一个人用脚轻敲"),

    # === UPPER BODY (61-70) ===
    ("a person bows", "一个人鞠躬"),
    ("a person nods their head", "一个人点头"),
    ("a person shakes their head", "一个人摇头"),
    ("a person turns around", "一个人转身"),
    ("a person looks to the left", "一个人向左看"),
    ("a person looks to the right", "一个人向右看"),
    ("a person bends forward", "一个人向前弯腰"),
    ("a person leans to the side", "一个人侧身倾斜"),
    ("a person twists their body", "一个人扭动身体"),
    ("a person shrugs their shoulders", "一个人耸肩"),

    # === COMPOUND/COMPLEX ACTIONS (71-85) ===
    ("a person walks forward then stops", "一个人向前走然后停下"),
    ("a person walks forward and waves", "一个人向前走并挥手"),
    ("a person sits down then stands up", "一个人坐下然后站起来"),
    ("a person jumps then walks forward", "一个人跳起来然后向前走"),
    ("a person picks something up from the ground", "一个人从地上捡东西"),
    ("a person puts something down on the ground", "一个人把东西放在地上"),
    ("a person walks to a chair and sits down", "一个人走到椅子旁坐下"),
    ("a person stands up and walks away", "一个人站起来走开"),
    ("a person turns around and walks back", "一个人转身走回去"),
    ("a person kicks a ball", "一个人踢球"),
    ("a person dribbles a basketball", "一个人运篮球"),
    ("a person swings a bat", "一个人挥棒"),
    ("a person drinks from a cup", "一个人用杯子喝水"),
    ("a person opens a door", "一个人开门"),
    ("a person closes a door", "一个人关门"),

    # === STYLE/MANNER VARIATIONS (86-100) ===
    ("a person walks happily", "一个人开心地走"),
    ("a person walks sadly", "一个人伤心地走"),
    ("a person walks confidently", "一个人自信地走"),
    ("a person sneaks forward quietly", "一个人悄悄向前走"),
    ("a person stumbles while walking", "一个人走路时绊了一下"),
    ("a person dances", "一个人跳舞"),
    ("a person stretches their body", "一个人伸展身体"),
    ("a person does jumping jacks", "一个人做开合跳"),
    ("a person exercises", "一个人锻炼"),
    ("a person warms up before exercise", "一个人运动前热身"),
    ("a person cools down after exercise", "一个人运动后放松"),
    ("a person practices martial arts", "一个人练武术"),
    ("a person does a push up", "一个人做俯卧撑"),
    ("a person does a squat", "一个人做深蹲"),
    ("a person takes a deep breath", "一个人深呼吸"),
]

# Save as JSON for easy loading
output = {
    "english": [p[0] for p in prompts],
    "chinese": [p[1] for p in prompts],
    "pairs": [{"id": i, "en": p[0], "zh": p[1]} for i, p in enumerate(prompts)]
}

with open("eval_prompts_100.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Generated {len(prompts)} prompt pairs")
print(f"Saved to eval_prompts_100.json")

# Also save as plain text for easy viewing
with open("eval_prompts_100.txt", "w", encoding="utf-8") as f:
    for i, (en, zh) in enumerate(prompts):
        f.write(f"{i+1:3d}. EN: {en}\n")
        f.write(f"     ZH: {zh}\n\n")

print(f"Saved readable version to eval_prompts_100.txt")
