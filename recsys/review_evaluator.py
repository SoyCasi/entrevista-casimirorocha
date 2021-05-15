from ReviewAnalysis import TextPreprocessor, ReviewAnalyzer
import sys
import traceback
import pickle

try:
    tp = TextPreprocessor()
    ra = ReviewAnalyzer()
    review = " ".join(sys.argv[0:])
    cleaned_review = tp.cleaner(review)

    if not cleaned_review:

        raise RuntimeError(
            "La informacion tomada de las review es muy pequeña y no parece ser los suficientemente valiosa como para ser valorada"
        )
    print(cleaned_review)
    infile = open("Data/recsys", "rb")
    info = pickle.load(infile)
    freqs = info["freqs"]
    theta = info["theta"]
    infile.close()
    y_score = ra.predict_review(cleaned_review, freqs, theta)
    if y_score > 0.5:
        text_score = "buena"
    else:
        text_score = "mala"
    print(text_score, y_score)
except Exception as err:
    print(f"Error  {err.args} \n {traceback.format_exc()}")
