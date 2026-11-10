from fakes import FakeClient, message, text

from claude_api_lab import evals, prompting


def test_builder_orders_sections_and_wraps_inputs_in_xml():
    prompt = (
        prompting.PromptBuilder(task="Summarize the report.")
        .context("report", "Revenue $4M")
        .guideline("Be brief.")
        .step("Read it.")
        .example("in", "out", why="short")
        .build()
    )
    assert prompt.startswith("Summarize the report.")
    assert "<report>\nRevenue $4M\n</report>" in prompt
    assert prompt.index("Guidelines") < prompt.index("Follow these steps") < prompt.index("<example>")
    assert "<why_ideal>\nshort\n</why_ideal>" in prompt


def test_engineered_doc_prompt_applies_every_technique():
    p = prompting.doc_qa_prompt_v2("What was revenue?", "Revenue was $1.")
    for marker in ("<document>", "<question>", "Guidelines:", "Follow these steps:", "<example>"):
        assert marker in p
    assert prompting.doc_qa_prompt_v1("Q", "D") == "D\n\nQ"


def test_xml_escapes_attributes():
    assert prompting.xml("doc", "x", source='a"b') == '<doc source="a&quot;b">\nx\n</doc>'


def test_code_grader_accepts_valid_and_rejects_invalid():
    assert evals.grade_by_code('{"a": 1}', "json") == 10
    assert evals.grade_by_code("{a: 1}", "json") == 0
    assert evals.grade_by_code("```python\nx = 1\n```", "python") == 10
    assert evals.grade_by_code("def (", "python") == 0
    assert evals.grade_by_code(r"^\d+$", "regex") == 10
    assert evals.grade_by_code("([", "regex") == 0
    assert evals.grade_by_code("anything", "text") == 10


def test_run_eval_averages_code_and_model_scores(tmp_path):
    case = evals.EvalCase(task="Return JSON with key a", format="json", solution_criteria="has key a")
    dataset = evals.Dataset(cases=[case])
    path = tmp_path / "nested" / "d.json"
    evals.save_dataset(dataset, path)
    assert evals.load_dataset(path) == dataset

    grade = evals.ModelGrade(strengths=["ok"], weaknesses=[], reasoning="fine", score=6)
    client = FakeClient(message(text('{"a": 1}')), grade)
    report = evals.run_eval(client, dataset, evals.engineered_prompt, concurrency=1)
    assert report.average == 8.0  # (10 code + 6 model) / 2
    assert report.to_dict()["results"][0]["reasoning"] == "fine"


def test_generate_dataset_uses_structured_output():
    ds = evals.Dataset(cases=[evals.EvalCase(task="t", format="regex", solution_criteria="c")])
    client = FakeClient(ds)
    assert evals.generate_dataset(client, "regex", 1) == ds
    assert client.last()["output_format"] is evals.Dataset
