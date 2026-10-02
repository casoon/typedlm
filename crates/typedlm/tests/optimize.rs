#![cfg(feature = "optimize")]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use typedlm::eval::{Dataset, ExactMatch, Verdict};
use typedlm::optimize::{
    FewShot, OptimizeError, bootstrap_demonstrations, demonstrations_from_labels, optimize_few_shot,
};
use typedlm::prelude::*;
use typedlm::testing::FnProvider;
use typedlm::{ProviderError, Request};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
enum Category {
    Billing,
    Technical,
    Shipping,
    Other,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Routing {
    category: Category,
}

/// Route a ticket to a team.
#[derive(TypedLm, Serialize, Deserialize, Clone, Debug)]
#[lm(output = Routing)]
struct Route {
    text: String,
}

const CATEGORIES: [&str; 4] = ["Billing", "Technical", "Shipping", "Other"];

/// Answers a ticket correctly only when a demonstration of its category was sent;
/// otherwise falls back to "Other". Ticket texts start with their category.
fn learner(request: &Request) -> Result<String, ProviderError> {
    let text = request.input["text"].as_str().unwrap();
    let wanted = CATEGORIES.iter().find(|c| text.starts_with(*c)).unwrap();
    let shown = request
        .demonstrations
        .iter()
        .any(|d| d.output["category"] == **wanted);
    let answer = if shown { wanted } else { &"Other" };
    Ok(format!(r#"{{"category": "{answer}"}}"#))
}

/// `per_category` labelled tickets per category.
fn dataset(per_category: usize, offset: usize) -> Dataset<Route> {
    let lines: Vec<String> = CATEGORIES
        .iter()
        .flat_map(|c| (0..per_category).map(move |i| (c, i + offset)))
        .map(|(c, i)| {
            format!(
                r#"{{"input": {{"text": "{c} ticket {i}"}}, "expected": {{"category": "{c}"}}}}"#
            )
        })
        .collect();
    Dataset::from_jsonl_str(&lines.join("\n")).unwrap()
}

#[tokio::test]
async fn finds_demonstrations_covering_every_category() {
    let provider = FnProvider::new(learner);
    let program = Program::<Route, _>::new(&provider).instructions("Route the ticket.");
    let pool = demonstrations_from_labels(&dataset(2, 0));
    assert_eq!(pool.len(), 8);
    let validation = dataset(5, 100);

    let options = FewShot {
        k: 4,
        max_calls: 2_000,
        seed: 3,
        concurrency: 4,
    };
    let result = optimize_few_shot(&program, &pool, &validation, &ExactMatch, options)
        .await
        .unwrap();

    assert!(
        (result.baseline.score - 0.25).abs() < 1e-9,
        "only Other is right without demos"
    );
    assert_eq!(result.best.score, 1.0);
    assert_eq!(result.comparison.verdict, Verdict::Improvement);
    assert!(result.local_optimum);

    let covered: Vec<&str> = result
        .compiled
        .demonstrations
        .iter()
        .map(|d| d.output["category"].as_str().unwrap())
        .collect();
    // "Other" is the fallback answer and needs no demonstration.
    for category in ["Billing", "Technical", "Shipping"] {
        assert!(covered.contains(&category), "{covered:?}");
    }
    assert_eq!(
        result.compiled.instructions.as_deref(),
        Some("Route the ticket.")
    );
    assert_eq!(result.compiled.provenance.as_ref().unwrap().score, 1.0);

    let accepted: Vec<f64> = result
        .trials
        .iter()
        .filter(|t| t.accepted)
        .map(|t| t.score)
        .collect();
    assert!(
        accepted.windows(2).all(|w| w[0] < w[1]),
        "accepted scores rise: {accepted:?}"
    );

    // The compiled program reproduces the best score.
    let restored = result.compiled.program::<Route, _>(&provider).unwrap();
    let report = typedlm::eval::evaluate(&restored, &validation, &ExactMatch, 4).await;
    assert_eq!(report.score, 1.0);
}

#[tokio::test]
async fn budget_limits_the_number_of_trials() {
    let provider = FnProvider::new(learner);
    let program = Program::<Route, _>::new(&provider);
    let pool = demonstrations_from_labels(&dataset(2, 0));
    let validation = dataset(5, 100); // 20 calls per evaluation

    let options = FewShot {
        k: 4,
        max_calls: 60,
        seed: 3,
        concurrency: 4,
    };
    let result = optimize_few_shot(&program, &pool, &validation, &ExactMatch, options)
        .await
        .unwrap();
    assert_eq!(result.trials.len(), 3);

    let err = optimize_few_shot(
        &program,
        &pool,
        &validation,
        &ExactMatch,
        FewShot {
            max_calls: 10,
            ..options
        },
    )
    .await
    .unwrap_err();
    assert!(
        matches!(
            err,
            OptimizeError::BudgetTooSmall {
                per_evaluation: 20,
                ..
            }
        ),
        "{err}"
    );
    let err = optimize_few_shot(&program, &pool[..2], &validation, &ExactMatch, options)
        .await
        .unwrap_err();
    assert_eq!(err, OptimizeError::PoolTooSmall { pool: 2, k: 4 });
}

#[tokio::test]
async fn bootstrapped_demonstrations_keep_only_correct_answers() {
    // Teacher knows Billing and Technical, so only those tickets become candidates.
    let teacher = FnProvider::new(|request: &Request| {
        let text = request.input["text"].as_str().unwrap();
        let answer = ["Billing", "Technical"]
            .into_iter()
            .find(|c| text.starts_with(c))
            .unwrap_or("Other");
        Ok(format!(r#"{{"category": "{answer}"}}"#))
    });
    let train = dataset(2, 0);
    let pool =
        bootstrap_demonstrations(&Program::<Route, _>::new(teacher), &train, &ExactMatch).await;
    let categories: Vec<&str> = pool
        .iter()
        .map(|d| d.output["category"].as_str().unwrap())
        .collect();
    assert_eq!(
        categories,
        [
            "Billing",
            "Billing",
            "Technical",
            "Technical",
            "Other",
            "Other"
        ]
    );
}

#[test]
fn partial_labels_are_not_candidates() {
    let dataset = Dataset::<Route>::from_jsonl_str(
        r#"{"input": {"text": "Billing a"}, "expected": {"category": "Billing"}}
{"input": {"text": "Billing b"}, "expected": {}}"#,
    )
    .unwrap();
    assert_eq!(demonstrations_from_labels(&dataset).len(), 1);
}

mod instructions {
    use std::sync::Mutex;

    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use typedlm::eval::{Dataset, ExactMatch, Verdict};
    use typedlm::optimize::{OptimizeError, Reflective, optimize_instructions};
    use typedlm::prelude::*;
    use typedlm::testing::FnProvider;
    use typedlm::{ProviderError, Request};

    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
    enum Urgency {
        Low,
        High,
    }

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct Classification {
        urgency: Urgency,
    }

    /// Classify a support ticket by urgency.
    #[derive(TypedLm, Serialize, Deserialize, Clone, Debug)]
    #[lm(output = Classification)]
    struct Classify {
        text: String,
    }

    const RULE: &str = "Outages are High.";

    /// Calls an outage High only when the instructions say so.
    fn student(request: &Request) -> Result<String, ProviderError> {
        let outage = request.input["text"].as_str().unwrap().contains("down");
        let knows = request.instructions.contains(RULE);
        let urgency = if outage && knows { "High" } else { "Low" };
        Ok(format!(r#"{{"urgency": "{urgency}"}}"#))
    }

    /// 10 tickets, half of them outages.
    fn validation() -> Dataset<Classify> {
        let lines: Vec<String> = (0..10)
            .map(|i| {
                let (text, urgency) =
                    if i % 2 == 0 { ("service down", "High") } else { ("question", "Low") };
                format!(r#"{{"input": {{"text": "{text} {i}"}}, "expected": {{"urgency": "{urgency}"}}}}"#)
            })
            .collect();
        Dataset::from_jsonl_str(&lines.join("\n")).unwrap()
    }

    #[tokio::test]
    async fn teacher_fixes_the_instructions_from_mistakes() {
        let seen = Mutex::new(Vec::new());
        let teacher = FnProvider::new(|request: &Request| {
            let mistakes = request.input["mistakes"].as_array().unwrap();
            seen.lock().unwrap().push(mistakes.len());
            assert!(
                mistakes
                    .iter()
                    .all(|m| m["expected"]["urgency"] == "High" && m["actual"]["urgency"] == "Low")
            );
            Ok(
                serde_json::json!({ "instructions": format!("Classify urgency. {RULE}") })
                    .to_string(),
            )
        });
        let provider = FnProvider::new(student);
        let program = Program::<Classify, _>::new(&provider);

        let result = optimize_instructions(
            &program,
            &teacher,
            &validation(),
            &ExactMatch,
            Reflective::default(),
        )
        .await
        .unwrap();
        assert_eq!(result.baseline.score, 0.5);
        assert_eq!(result.best.score, 1.0);
        assert_eq!(result.comparison.verdict, Verdict::Improvement);
        assert_eq!(result.trials.len(), 1, "stops once nothing is wrong");
        assert!(result.trials[0].accepted);
        assert!(
            result
                .compiled
                .instructions
                .as_deref()
                .unwrap()
                .contains(RULE)
        );
        assert_eq!(*seen.lock().unwrap(), [5]);

        let restored = result.compiled.program::<Classify, _>(&provider).unwrap();
        assert_eq!(
            typedlm::eval::evaluate(&restored, &validation(), &ExactMatch, 2)
                .await
                .score,
            1.0
        );
    }

    #[tokio::test]
    async fn useless_proposals_keep_the_original() {
        let shown = Mutex::new(Vec::new());
        let teacher = FnProvider::new(|request: &Request| {
            let first = request.input["mistakes"][0]["input"]["text"]
                .as_str()
                .unwrap()
                .to_string();
            shown.lock().unwrap().push(first);
            Ok(r#"{"instructions": "Be careful."}"#.into())
        });
        let provider = FnProvider::new(student);
        let program = Program::<Classify, _>::new(&provider);
        let options = Reflective {
            iterations: 3,
            mistakes_shown: 2,
            ..Reflective::default()
        };

        let result = optimize_instructions(&program, &teacher, &validation(), &ExactMatch, options)
            .await
            .unwrap();
        assert_eq!(result.trials.len(), 3);
        assert!(
            result
                .trials
                .iter()
                .all(|t| !t.accepted && t.score == Some(0.5))
        );
        assert_eq!(
            result.compiled.instructions, None,
            "the original instructions stay"
        );
        assert_eq!(result.comparison.verdict, Verdict::NoSignificantChange);
        let shown = shown.lock().unwrap();
        assert_ne!(
            shown[0], shown[1],
            "each proposal sees other mistakes: {shown:?}"
        );
    }

    #[tokio::test]
    async fn failed_proposals_are_recorded_and_budget_is_checked() {
        let teacher = FnProvider::new(|_: &Request| Ok("no json".into()));
        let provider = FnProvider::new(student);
        let program = Program::<Classify, _>::new(&provider).max_repairs(0);
        let options = Reflective {
            iterations: 2,
            ..Reflective::default()
        };

        let result = optimize_instructions(&program, &teacher, &validation(), &ExactMatch, options)
            .await
            .unwrap();
        assert_eq!(result.trials.len(), 2);
        assert!(
            result
                .trials
                .iter()
                .all(|t| t.score.is_none() && t.error.is_some())
        );

        let small = Reflective {
            max_calls: 15,
            ..Reflective::default()
        };
        let err = optimize_instructions(&program, &teacher, &validation(), &ExactMatch, small)
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            OptimizeError::BudgetTooSmall {
                per_evaluation: 10,
                ..
            }
        ));
    }
}
