"""One-time authoring of fixed input/output files from bundled source texts."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def excerpt(number, start, end):
    text = (ROOT / "sources" / f"pg{number}.txt").read_text(encoding="utf-8")
    first = re.search(start, text, re.MULTILINE)
    if not first:
        raise ValueError(start)
    last = re.search(end, text[first.end():], re.MULTILINE)
    if not last:
        raise ValueError(end)
    return text[first.start():first.end() + last.start()].strip()


def qa(question, answer, evidence, aliases=()):
    return {"question": question, "answer": answer, "aliases": list(aliases), "evidence": evidence}


definitions = [
    ("alice-01", 11, "Alice's Adventures in Wonderland", "Lewis Carroll", [
        (r"^CHAPTER I\.\n", r"^CHAPTER III\.\n", "ALICE-A-071", [
            qa("What animal runs close to Alice near the beginning?", "White Rabbit", "White Rabbit with pink eyes", ["a white rabbit", "the white rabbit"]),
            qa("What colour are that animal's eyes?", "pink", "White Rabbit with pink eyes"),
        ]),
        (r"^CHAPTER III\.\n", r"^CHAPTER V\.\n", "ALICE-B-492", [
            qa("Which bird proposes the race as a way to get dry?", "Dodo", "said the Dodo in an offended tone", ["the dodo"]),
            qa("What does that bird call the race?", "Caucus-race", "the best thing to get us dry would be a Caucus-race", ["a caucus race", "caucus race"]),
        ]),
        (r"^CHAPTER V\.\n", r"^CHAPTER VII\.\n", "ALICE-C-835", [
            qa("What smoking implement does the Caterpillar take out of its mouth?", "hookah", "the Caterpillar took the hookah out of its mouth", ["a hookah", "the hookah"]),
            qa("What short instruction does the Caterpillar give after calling Alice back?", "Keep your temper", "Keep your temper"),
        ]),
    ]),
    ("oz-02", 55, "The Wonderful Wizard of Oz", "L. Frank Baum", [
        (r"^Chapter I\n", r"^Chapter III\n", "OZ-A-263", [
            qa("In which US state does Dorothy live at the start?", "Kansas", "great Kansas prairies"),
            qa("What is the name of Dorothy's dog?", "Toto", "Toto that made Dorothy laugh"),
        ]),
        (r"^Chapter III\n", r"^Chapter V\n", "OZ-B-914", [
            qa("What two colours make up the checks on Dorothy's clean gingham dress?", "white and blue", "checks of white and blue", ["blue and white"]),
            qa("What colour is the sunbonnet she ties on her head?", "pink", "tied her pink sunbonnet on her head"),
        ]),
        (r"^Chapter V\n", r"^Chapter VII\n", "OZ-C-506", [
            qa("What material is the motionless woodman made entirely of?", "tin", "a man made entirely of tin"),
            qa("What does the woodman ask Dorothy to fetch to free his joints?", "oil-can", "Get an oil-can and oil my joints", ["an oil can", "oil can", "an oil-can"]),
        ]),
    ]),
    ("treasure-03", 120, "Treasure Island", "Robert Louis Stevenson", [
        (r"^I\nThe Old Sea-dog", r"^III\nThe Black Spot", "SEA-A-648", [
            qa("What is the name of the inn kept by the narrator's father?", "Admiral Benbow", "my father kept the Admiral Benbow inn", ["admiral benbow inn", "the admiral benbow"]),
            qa("What kind of chest follows the old seaman to the inn?", "sea-chest", "his sea-chest following behind him", ["sea chest", "a sea chest", "a sea-chest"]),
        ]),
        (r"^III\nThe Black Spot", r"^VI\nThe Captain", "SEA-B-185", [
            qa("What alcoholic drink does the captain plead with Jim to bring?", "rum", "one noggin of rum", ["a noggin of rum"]),
            qa("What gold coin does he offer Jim for a noggin?", "golden guinea", "a golden guinea for a noggin", ["a golden guinea", "guinea", "a guinea", "gold guinea"]),
        ]),
        (r"^VI\nThe Captain", r"^VIII\n", "SEA-C-729", [
            qa("Who opens the door when Jim knocks at Dr. Livesey's house?", "maid", "The door was opened almost at once by the maid", ["the maid", "a maid"]),
            qa("In which room does the servant show the visitors the squire and doctor?", "library", "a great library, all lined with bookcases", ["the library", "a library", "great library"]),
        ]),
    ]),
    ("holmes-04", 1661, "The Adventures of Sherlock Holmes", "Arthur Conan Doyle", [
        (r"^I\. A SCANDAL IN BOHEMIA\n", r"^II\. THE RED-HEADED LEAGUE\n", "CASE-A-397", [
            qa("What is the full name of the woman Holmes calls 'the woman'?", "Irene Adler", "the late Irene Adler"),
            qa("On which street are Holmes's lodgings?", "Baker Street", "our lodgings in Baker Street"),
        ]),
        (r"^II\. THE RED-HEADED LEAGUE\n", r"^III\. A CASE OF IDENTITY\n", "CASE-B-652", [
            qa("What is the full name of the portly red-haired client?", "Jabez Wilson", "Mr. Jabez Wilson here", ["mr jabez wilson", "mr. jabez wilson"]),
            qa("In which season does Watson call on Holmes at the start of this story?", "autumn", "one day in the autumn of last year", ["fall"]),
        ]),
        (r"^III\. A CASE OF IDENTITY\n", r"^IV\. THE BOSCOMBE VALLEY MYSTERY\n", "CASE-C-048", [
            qa("What gemstone is in the centre of Holmes's snuffbox lid?", "amethyst", "a great amethyst in the centre of the lid", ["an amethyst"]),
            qa("Which monarch gave Holmes that snuffbox?", "King of Bohemia", "a little souvenir from the King of Bohemia", ["the king of bohemia"]),
        ]),
    ]),
]


def normal(text):
    return " ".join(text.split()).casefold()


if __name__ == "__main__":
    cases, answers = [], {}
    (ROOT / "inputs_source").mkdir(exist_ok=True)
    (ROOT / "expected").mkdir(exist_ok=True)
    for case_id, number, title, author, parts in definitions:
        sections, expected = [], {"document_id": case_id}
        for i, (start, end, code, questions) in enumerate(parts, 1):
            text = excerpt(number, start, end)
            for question in questions:
                if normal(question["evidence"]) not in normal(text):
                    raise ValueError(f"Missing evidence {case_id}: {question['evidence']}")
            sections.append({"code": code, "text": text, "questions": questions})
            expected[f"excerpt_{i}_code"] = code
            for j, question in enumerate(questions):
                expected[f"answer_{(i - 1) * 2 + j + 1}"] = question["answer"]
        case = {"id": case_id, "title": title, "author": author,
                "source_url": f"https://www.gutenberg.org/ebooks/{number}", "sections": sections}
        cases.append(case)
        answers[case_id] = expected
        lines = [f"Document ID: {case_id}", f"{title} — {author}",
                 "Read the following three excerpts; some passages are omitted between them."]
        for i, section in enumerate(sections, 1):
            lines += [f"\nEXCERPT {i}; CODE {section['code']}\n", section["text"], "\nEND OF EXCERPT"]
        lines += ["\nQUESTIONS: return one JSON object with document_id, excerpt_1_code, "
                  "excerpt_2_code, excerpt_3_code, and answer_1 through answer_6. "
                  "Use short English strings. No explanation; do not continue the dialogue."]
        for i, section in enumerate(sections):
            for j, question in enumerate(section["questions"]):
                lines.append(f"answer_{i * 2 + j + 1} (excerpt {i + 1}): {question['question']}")
        (ROOT / "inputs_source" / f"{case_id}.txt").write_text("\n".join(lines), encoding="utf-8")
        (ROOT / "expected" / f"{case_id}.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")
    (ROOT / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "expected" / "bs4.json").write_text(json.dumps(answers, indent=2), encoding="utf-8")
    print("Wrote four fixed documents and 40 evidence-checked expected fields.")
