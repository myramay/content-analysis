"""
train.py — ED/Skinnytok harmful content classifier
Trains a TF-IDF + Logistic Regression model with:
  - Domain lexicon boost features
  - VADER sentiment features
  - 200 synthetic labeled examples (SAFE / WARN / HARMFUL)
"""

import os
import re
import numpy as np
import scipy.sparse as sp
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Domain lexicon — terms associated with ED content
# Short single-word terms use word-boundary matching; multi-word use substring.
# ---------------------------------------------------------------------------
LEXICON = [
    "thinspo", "thinspiration", "restrict", "restricting", "restriction",
    "fast", "fasting", "bones", "skinny", "hwp", "sw", "gw", "cw", "lw", "hw",
    "deficit", "purge", "purging", "skip", "starve", "starving",
    "ana", "mia", "ednos", "thigh gap", "collarbone", "collarbones",
    "ribs", "spine", "hipbones", "hip bones", "diet pills", "laxative",
    "laxatives", "binge", "pro-ana", "pro ana", "body check", "bodycheck",
    "skinnytok", "waist check", "bone rattle",
]

# ---------------------------------------------------------------------------
# Synthetic dataset — 200 labeled examples
# SAFE: normal food/lifestyle content, body positivity
# WARN: diet talk, calorie counting, mild weight loss language
# HARMFUL: explicit ED content, restriction, thinspo, purging
# ---------------------------------------------------------------------------
DATA = [
    # ── SAFE (70) ────────────────────────────────────────────────────────────
    ("I made the most amazing homemade pizza with fresh basil and buffalo mozzarella tonight!", "SAFE"),
    ("Just got back from a 5K run, feeling so energized and proud of myself!", "SAFE"),
    ("Tried a new Thai restaurant downtown, the pad thai was absolutely incredible", "SAFE"),
    ("Made avocado toast with poached eggs this morning, starting the day right", "SAFE"),
    ("Family BBQ today with all the classic summer foods, so much fun", "SAFE"),
    ("Baked sourdough bread from scratch for the first time, it turned out perfectly", "SAFE"),
    ("New smoothie: banana, spinach, almond milk, and a spoonful of peanut butter", "SAFE"),
    ("Visited a farmers market and came home with so many gorgeous fresh veggies", "SAFE"),
    ("Cooked a big meal prep batch for the week: rice, roasted veggies, and chicken", "SAFE"),
    ("The best chocolate lava cake I have ever tasted, highly recommend this bakery", "SAFE"),
    ("Morning yoga session complete! My flexibility has improved so much lately", "SAFE"),
    ("Going on a hike this weekend, can't wait to enjoy nature and get some movement in", "SAFE"),
    ("Tried a new dance class today and it was the most fun workout I have had in ages", "SAFE"),
    ("Swimming laps at the community pool is my favorite summer activity", "SAFE"),
    ("Just finished a really challenging rock climbing session, my forearms are done", "SAFE"),
    ("All bodies are beautiful and worthy of respect, period. No exceptions.", "SAFE"),
    ("Self-care Sunday: face mask, a good book, and a cup of chamomile tea", "SAFE"),
    ("Loving my body for everything it can do, not just how it looks", "SAFE"),
    ("Reminder: your weight does not determine your worth as a person", "SAFE"),
    ("Body neutrality is about appreciating what your body can do for you", "SAFE"),
    ("Had the best brunch with friends today, mimosas and eggs benedict all around", "SAFE"),
    ("Dinner party tonight! Making my grandma's famous lasagna recipe from scratch", "SAFE"),
    ("Weekend trip to the coast, ate the freshest seafood I have ever had in my life", "SAFE"),
    ("Trying out a new vegetarian recipe: chickpea curry with homemade naan bread", "SAFE"),
    ("Farmers market haul! So many beautiful heirloom tomatoes this time of year", "SAFE"),
    ("Went to a cooking class and learned how to make fresh pasta from scratch", "SAFE"),
    ("The local bakery just released a new seasonal flavor and it is absolutely phenomenal", "SAFE"),
    ("Birthday cake reveal: three layers of red velvet with cream cheese frosting!", "SAFE"),
    ("Girls trip was amazing, we ate our way through every restaurant in the city", "SAFE"),
    ("Homemade ramen from scratch took all day but was completely and utterly worth it", "SAFE"),
    ("Playing with my dog at the park, best stress relief there is", "SAFE"),
    ("Reading a new book in the sun with a cold glass of lemonade, perfect afternoon", "SAFE"),
    ("Concert was incredible tonight, danced for three hours straight", "SAFE"),
    ("Movie night with homemade popcorn and all the snacks, living my best life", "SAFE"),
    ("Volunteering at the community garden this morning was so rewarding", "SAFE"),
    ("Meditation practice has helped my focus and sleep so much lately", "SAFE"),
    ("Just finished a great novel, 10/10 would recommend to absolutely everyone", "SAFE"),
    ("Art class was so therapeutic today, I forgot how much I love painting", "SAFE"),
    ("Sunset hike with amazing views, grateful for healthy legs that carried me up", "SAFE"),
    ("Coffee shop working session: oat milk latte and a slice of banana bread", "SAFE"),
    ("My doctor said all my health markers look great at my annual checkup!", "SAFE"),
    ("Celebrating my friend's birthday with a big dinner tonight, can't wait!", "SAFE"),
    ("Took a rest day today and my body really needed it, recovery is part of fitness", "SAFE"),
    ("New running shoes finally broke in, my feet are incredibly happy", "SAFE"),
    ("Kids baked cookies today, kitchen is a disaster but we had so much fun", "SAFE"),
    ("Sunday morning pancakes are a sacred tradition in our house, wouldn't skip it", "SAFE"),
    ("Taco Tuesday! Homemade carnitas with all the toppings, absolute perfection", "SAFE"),
    ("Just discovered the best noodle place in my neighborhood, going back tomorrow", "SAFE"),
    ("Made my first attempt at croissants and they were shockingly almost perfect", "SAFE"),
    ("Grilling season is here, marinated chicken and corn on the cob tonight", "SAFE"),
    ("Hiking trail snacks: trail mix, apple slices, and a peanut butter sandwich", "SAFE"),
    ("Morning swim was so peaceful, just me and the sunrise on the water", "SAFE"),
    ("Gym session done! Loving the strength gains I have made this year", "SAFE"),
    ("Pilates class was challenging but in the absolute best way possible", "SAFE"),
    ("Got into cycling recently, did my first 20-mile ride this morning!", "SAFE"),
    ("Celebrating all the ways my body keeps me healthy and strong every day", "SAFE"),
    ("Finding joy in movement that feels good rather than punishing or painful", "SAFE"),
    ("Health at every size is a valid and important framework worth understanding", "SAFE"),
    ("Intuitive eating has genuinely changed my relationship with food for the better", "SAFE"),
    ("Every body deserves nourishment, rest, and unconditional care", "SAFE"),
    ("Date night pasta from scratch, romantic dinner at home is underrated", "SAFE"),
    ("Just picked up a CSA box and discovered so many new vegetables to try!", "SAFE"),
    ("The soup I made from the last garden tomatoes of the season tastes like summer", "SAFE"),
    ("Potluck dinner party: everyone brought something completely amazing", "SAFE"),
    ("Making my way through Julia Child's cookbook, learned to make bouillabaisse", "SAFE"),
    ("Ice cream sundae bar at the block party, three scoops please", "SAFE"),
    ("Breakfast in bed on a lazy Sunday: french toast with maple syrup and berries", "SAFE"),
    ("Hot chocolate and a fire on a cold winter night, pure cozy perfection", "SAFE"),
    ("Lunch with my grandma at her favorite diner, she always orders the same thing", "SAFE"),
    ("Celebrated finishing a big project with my favorite fancy dinner out tonight", "SAFE"),

    # ── WARN (65) ────────────────────────────────────────────────────────────
    ("Trying to cut back on carbs this month to lose a few pounds before the trip", "WARN"),
    ("Only had 1300 calories today but feeling pretty good about staying on track", "WARN"),
    ("Does anyone else track macros? I am trying to hit 130g protein daily right now", "WARN"),
    ("My goal is to lose 15 pounds before summer, starting my diet today", "WARN"),
    ("Skipped breakfast again, just wasn't hungry in the morning honestly", "WARN"),
    ("Using MyFitnessPal to track everything I eat, really eye-opening numbers", "WARN"),
    ("Trying CICO (calories in calories out) method to drop some weight", "WARN"),
    ("Cut out sugar and processed foods this month and feeling noticeably different", "WARN"),
    ("Drinking more water throughout the day to help curb appetite between meals", "WARN"),
    ("Low carb dinner tonight: grilled chicken and salad, skipping the rice tonight", "WARN"),
    ("Anyone else doing a clean eating challenge? Day 5 of no added sugar", "WARN"),
    ("Before: 165 lbs. Goal: 145 lbs by my birthday in October. Starting now.", "WARN"),
    ("Having a planned cheat meal today after a week of clean eating, I earned it", "WARN"),
    ("Joining a weight loss support group for accountability, really helps me", "WARN"),
    ("Tried a detox tea and it actually helped me feel less bloated this week", "WARN"),
    ("Weighing myself every morning to keep myself accountable to my goals", "WARN"),
    ("Low calorie dinner tonight: zucchini noodles with marinara, only 280 calories", "WARN"),
    ("Cutting back on alcohol to help with my overall weight loss goals this year", "WARN"),
    ("Swapping rice for cauliflower rice to reduce carbs in my evening meals", "WARN"),
    ("I have been in a 500 calorie deficit this week and already seeing results", "WARN"),
    ("Intermittent fasting 16:8, two weeks in and I love the structure it gives me", "WARN"),
    ("Body recomposition: trying to lose body fat while maintaining muscle mass", "WARN"),
    ("Progress pic: down 8 lbs in 6 weeks so far, slow and steady wins the race", "WARN"),
    ("Meal prepping low calorie lunches for the whole week ahead", "WARN"),
    ("Going to the gym twice a day this week to speed up my current cut phase", "WARN"),
    ("Dropped a size in jeans! Feel like I am finally making real progress", "WARN"),
    ("Diet starts Monday for real this time, need to lose these holiday pounds", "WARN"),
    ("Anyone tried the keto diet? I am seriously thinking about giving it a shot", "WARN"),
    ("Trying to eat at 1500 calories to lose weight but it is honestly challenging", "WARN"),
    ("Scale hasn't moved in a week, frustrating but staying the course regardless", "WARN"),
    ("Thinking about trying paleo, does anyone have real experience with it?", "WARN"),
    ("Ate too much at dinner tonight, feeling guilty so going for an extra walk", "WARN"),
    ("Looking up lower calorie versions of all my favorite comfort food recipes", "WARN"),
    ("Switched to diet soda and sugar-free options everywhere to help weight loss", "WARN"),
    ("Summer body prep: starting my cut 12 weeks out from my beach vacation", "WARN"),
    ("My dietitian put me on a structured 1400 calorie plan, working with a pro", "WARN"),
    ("Trying to reduce portions without obsessively counting every single calorie", "WARN"),
    ("Watching my sodium intake this week to try to reduce water weight retention", "WARN"),
    ("Substituting fruit for dessert whenever I can to cut back on empty calories", "WARN"),
    ("Yoga and walking are my low impact active recovery options on rest days", "WARN"),
    ("Avoiding eating after 7pm as part of my weight management routine lately", "WARN"),
    ("Reading food labels much more carefully now to make genuinely better choices", "WARN"),
    ("Replacing evening snacks with sparkling water and it really does help", "WARN"),
    ("Doing a 30 day fitness challenge with friends for mutual motivation and fun", "WARN"),
    ("Set a fitness goal for the year and checking in monthly on my progress", "WARN"),
    ("Weighed in this morning: down 2 lbs this week. Slow progress is still progress!", "WARN"),
    ("Cutting processed foods this month to observe how my body actually responds", "WARN"),
    ("Working with a trainer to build a caloric deficit that is genuinely sustainable", "WARN"),
    ("Trying apple cider vinegar before meals, heard it helps with metabolism", "WARN"),
    ("Decided to track food intake for a month to really understand my eating habits", "WARN"),
    ("I am so bloated today, definitely overdid the sodium this whole weekend", "WARN"),
    ("Clean eating week: no junk food, no alcohol, focusing entirely on whole foods", "WARN"),
    ("Meal timing matters to me: trying to eat the majority of calories early in day", "WARN"),
    ("Low glycemic index eating plan to help manage my blood sugar levels better", "WARN"),
    ("Body fat percentage is what I am really focused on, not just the scale number", "WARN"),
    ("Anyone have tips for reducing belly fat? Consistent cardio seems to help me", "WARN"),
    ("Counting points with Weight Watchers app, accountability really helps me stay on", "WARN"),
    ("Trying to eat more fiber and protein at each meal to stay fuller much longer", "WARN"),
    ("Week 3 of my weight loss journey, tracking every bite I put in my mouth", "WARN"),
    ("Before photo taken today, excited to see where I am in exactly 3 months", "WARN"),
    ("Skipped the birthday cake at work today, trying to stay disciplined this week", "WARN"),
    ("Portion control is actually harder for me than calorie counting honestly", "WARN"),
    ("Low calorie smoothie in the morning keeps me full until almost noon", "WARN"),
    ("On a cut right now, eating at a 300 calorie deficit for 8 more weeks total", "WARN"),
    ("Lost 5 lbs this month on a restricted diet, slow progress is still real progress", "WARN"),

    # ── HARMFUL (65) ─────────────────────────────────────────────────────────
    ("Thinspo goals for this summer, I want to look exactly like her so bad 💀✨", "HARMFUL"),
    ("SW: 145 CW: 127 GW: 105, almost halfway to my goal weight now!", "HARMFUL"),
    ("Day 4 of fasting, the hunger pangs are intense but worth it for the results", "HARMFUL"),
    ("How to restrict without your parents noticing: eat small amounts and move food around your plate", "HARMFUL"),
    ("Thinspiration: when your collarbones and hip bones are both visible at once", "HARMFUL"),
    ("Purging after my binge tonight, I feel so disgusted with myself for eating that", "HARMFUL"),
    ("Nothing tastes as good as skinny feels, this will always forever be my mantra", "HARMFUL"),
    ("Ana tips: always eat standing up and chew slowly to make the food last longer", "HARMFUL"),
    ("My goal is to have a thigh gap, doing 500 jumping jacks every single day", "HARMFUL"),
    ("Body check: can you see my ribs? Is this progress? I think I can finally see them", "HARMFUL"),
    ("Skip breakfast and lunch, then have a small dinner under 300 calories total", "HARMFUL"),
    ("Following a pro-ana diet plan I found online, 500 calories max per day strict", "HARMFUL"),
    ("Mia is the only one who understands me right now, nobody else gets it at all", "HARMFUL"),
    ("EDNOS tip: eat only when you feel like you are about to pass out from hunger", "HARMFUL"),
    ("My thighs still touch and I am disgusted by it. Must restrict more this entire week", "HARMFUL"),
    ("Thinspiration: her collarbones are so beautiful, I want that exact body so bad", "HARMFUL"),
    ("Fast for 3 days, then one tiny meal, then fast again. Repeat until goal weight", "HARMFUL"),
    ("HW: 160 LW: 98 CW: 112 GW: 95, I was so close and now I have failed myself", "HARMFUL"),
    ("If you eat 500 calories or less you lose weight much faster, simple math really", "HARMFUL"),
    ("The sound of my stomach growling means it is working, stay strong and keep going", "HARMFUL"),
    ("Triggered by how much I ate today, going to purge and start completely fresh tomorrow", "HARMFUL"),
    ("Looking for thinspo and diet buddies who really understand the struggle daily", "HARMFUL"),
    ("Soft restriction: only eat if you have burned more calories than you are about to consume", "HARMFUL"),
    ("Bones are beautiful, the visible clavicle is my ultimate and only aesthetic goal", "HARMFUL"),
    ("You do not need food, you need self-control. That is the only real truth here", "HARMFUL"),
    ("Ana diary day 12: only had black coffee and a small apple today, feeling proud", "HARMFUL"),
    ("Skipping meals is not disordered if it is just your lifestyle choice right", "HARMFUL"),
    ("I feel powerful when I say no to food that my body is desperately begging for", "HARMFUL"),
    ("The hunger is just your fat crying as it disappears forever, embrace that feeling", "HARMFUL"),
    ("Only 400 calories today because I have zero self control when I eat any more", "HARMFUL"),
    ("Thinspiration: she is so tiny and perfect, her body is absolute goals for life", "HARMFUL"),
    ("Water fasting day 5: lightheaded and weak but I can see my ribs so much better now", "HARMFUL"),
    ("My dietitian wants me to eat 1800 calories but that honestly seems way too much", "HARMFUL"),
    ("Diet pills and laxatives are the only things keeping me going right now honestly", "HARMFUL"),
    ("HW: 150 GW: 90 thinking about going even lower than that to be extra safe", "HARMFUL"),
    ("So hungry I cannot sleep but I feel like I completely deserve this suffering", "HARMFUL"),
    ("Body check: I can finally see my spine through my back clearly, real progress!", "HARMFUL"),
    ("SW goals: want to be able to wrap my entire hand all the way around my thigh", "HARMFUL"),
    ("If you are not hungry enough to eat an apple, you are not really hungry, skip it", "HARMFUL"),
    ("Fasting almost every day with one tiny meal per week, seeing absolutely amazing results", "HARMFUL"),
    ("The only thing that matters to me is the number on the scale going lower every day", "HARMFUL"),
    ("Purging does not count if it is just once right? I really needed to do it tonight", "HARMFUL"),
    ("My brain goes completely quiet when I am restricting, it is the only peace I have", "HARMFUL"),
    ("GW: to be the thinnest person in absolutely every single room I walk into", "HARMFUL"),
    ("Thinspo account update: dropped 3 more pounds this week, only 15 more pounds to go", "HARMFUL"),
    ("I ate one granola bar today and want to compensate by not eating anything tomorrow", "HARMFUL"),
    ("Size 00 is the only goal that matters, doing whatever it takes to get there", "HARMFUL"),
    ("Looking at thinspo pictures when I feel like eating, it kills the urge immediately", "HARMFUL"),
    ("Tips for hiding your weight loss from concerned family members who keep watching", "HARMFUL"),
    ("Restrict restrict restrict, that is the only path forward I can see for myself", "HARMFUL"),
    ("BMI of 16 is actually really healthy if you happen to have small bones naturally", "HARMFUL"),
    ("No food after noon, no food before dinner, and a tiny dinner only if absolutely needed", "HARMFUL"),
    ("Started laxative tea routine and it has cleared out so much, down 2 lbs overnight", "HARMFUL"),
    ("When people say I am too thin I take it as a compliment and push myself harder", "HARMFUL"),
    ("The goal is for my collarbones to look exactly like this when I am photographed", "HARMFUL"),
    ("Revenge body plan: restrict as hard as possible until he completely regrets leaving me", "HARMFUL"),
    ("Eating is giving up. Every single time I say no to food I win a small battle", "HARMFUL"),
    ("Body dysmorphia is just seeing clearly what everyone else pretends not to notice", "HARMFUL"),
    ("Pro-ana tips that actually work: ice water, black coffee, chewing and spitting food", "HARMFUL"),
    ("Skinnytok told me to eat one meal a day and it is the best advice I ever received", "HARMFUL"),
    ("My GW keeps getting lower the closer I get, I genuinely do not know when to stop", "HARMFUL"),
    ("The bones in my chest are finally starting to show through my skin, so so happy", "HARMFUL"),
    ("Only thing in my stomach today is black coffee and diet pills, keeping it together", "HARMFUL"),
    ("Deleted all the food from my apartment so I cannot possibly be tempted to binge", "HARMFUL"),
    ("Starving yourself is not a disorder if you are in control of it and choose it", "HARMFUL"),
]

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
analyzer = SentimentIntensityAnalyzer()


def lexicon_count(text: str) -> int:
    """Count lexicon term hits using word-boundary matching for single words."""
    text_lower = text.lower()
    count = 0
    for term in LEXICON:
        if " " in term or "-" in term:
            count += 1 if term in text_lower else 0
        else:
            count += 1 if re.search(r"\b" + re.escape(term) + r"\b", text_lower) else 0
    return count


def build_features(texts, vectorizer, fit=False):
    """Combine TF-IDF, lexicon count, and VADER sentiment into one feature matrix."""
    if fit:
        tfidf = vectorizer.fit_transform(texts)
    else:
        tfidf = vectorizer.transform(texts)

    lex = np.array([[lexicon_count(t)] for t in texts], dtype=np.float32)

    vader_rows = []
    for t in texts:
        s = analyzer.polarity_scores(t)
        vader_rows.append([s["neg"], s["neu"], s["pos"], s["compound"]])
    vader = np.array(vader_rows, dtype=np.float32)

    return sp.hstack([tfidf, sp.csr_matrix(lex), sp.csr_matrix(vader)])


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def main():
    texts, labels = zip(*DATA)
    texts, labels = list(texts), list(labels)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)

    X_train = build_features(X_train_raw, vectorizer, fit=True)
    X_test = build_features(X_test_raw, vectorizer, fit=False)

    model = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=sorted(set(labels))))

    print("=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(labels)))
    classes = sorted(set(labels))
    header = f"{'':>10}" + "".join(f"{c:>10}" for c in classes)
    print(header)
    for i, row in enumerate(cm):
        print(f"{classes[i]:>10}" + "".join(f"{v:>10}" for v in row))

    os.makedirs("model", exist_ok=True)
    bundle = {
        "vectorizer": vectorizer,
        "model": model,
        "lexicon": LEXICON,
    }
    joblib.dump(bundle, "model/detector.joblib")
    print("\nModel saved to model/detector.joblib")


if __name__ == "__main__":
    main()
