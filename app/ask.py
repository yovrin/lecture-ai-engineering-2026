"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --interactive

ツールを持たない「素の」呼び出し。対話モードでは会話履歴を保持する。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
from collections.abc import MutableSequence

from google import genai
from google.genai import errors
from google.genai import types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(
    question: str,
    history: MutableSequence[types.Content] | None = None,
    *,
    client: genai.Client | None = None,
) -> str:
    """質問を送り、応答を会話履歴に追加して返す。

    ``history`` を同じリストのまま次の呼び出しにも渡すことで、前の質問と
    応答を含めてモデルに送信できる。単発利用時の呼び出し方も維持する。
    """
    conversation = history if history is not None else []
    conversation.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))

    active_client = client or genai.Client(api_key=get_api_key())
    response = active_client.models.generate_content(
        model=MODEL,
        contents=list(conversation),
    )
    if not response.candidates or response.candidates[0].content is None:
        raise ValueError("モデルから有効な応答が返されませんでした")
    conversation.append(response.candidates[0].content)

    usage = response.usage_metadata
    print(f"[入力トークン数] {usage.prompt_token_count}")
    print(f"[出力トークン数] {usage.candidates_token_count}")

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="終了するまで質問を繰り返し、会話履歴を保持する",
    )
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.interactive:
            client = genai.Client(api_key=get_api_key())
            history: list[types.Content] = []
            print("質問を入力してください（終了するには空行または Ctrl-D）。\n")
            while True:
                try:
                    question = input("質問> ").strip()
                except EOFError:
                    print()
                    break
                if not question:
                    break
                print(f"\n[質問] {question}\n")
                print(ask(question, history, client=client))
                print()
        else:
            print(f"[質問] {args.question}\n")
            print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
