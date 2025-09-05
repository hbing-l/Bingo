# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from flask import Flask, request, jsonify
from llmlingua.prompt_compressor import PromptCompressor
import torch


compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,
    device_map="auto",
)


app = Flask(__name__)

@app.route("/compress", methods=["POST"])
def compress():
    data = request.get_json(force=True)
    text = data.get("text")
    rate = float(data.get("rate", 0.6))

    if not text:
        return jsonify({"error": "Field 'text' is required."}), 400
    if not (0 < rate <= 1):
        return jsonify({"error": "Field 'rate' must be in (0,1]."}), 400

    try:
        compressed = compressor.compress_prompt_llmlingua2([text], rate=rate)
        tmp_original_probs = []
        for p_list in compressed['original_probs']:
            tmp_p_list = []
            for p in p_list:
                tmp_p_list.append(float(p))
            tmp_original_probs.append(tmp_p_list)
        compressed['original_probs'] = tmp_original_probs
        return jsonify(**compressed)
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1234, debug=False)