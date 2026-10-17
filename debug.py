from langchain_core.messages import HumanMessage

from agent import graph


if __name__ == "__main__":
    question = input("Question: ").strip()

    with open("debug_output.txt", "w", encoding="utf-8") as f:
        f.write(f"Question: {question}\n")
        for update in graph.stream(
            {"messages": [HumanMessage(content=question)]},
            stream_mode="updates",
        ):
            for node_name, changes in update.items():
                label = f"\n--- Node: {node_name} ---\n"
                print(label)
                f.write(label)

                for message in changes.get("messages", []):
                    formatted = message.pretty_repr()
                    print(formatted)
                    f.write(formatted + "\n")
                f.flush()
