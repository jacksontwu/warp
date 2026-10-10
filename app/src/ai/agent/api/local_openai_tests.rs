use super::*;

#[test]
fn builds_chat_completions_url_from_v1_base() {
    let config = LocalOpenAIConfig::from_values(
        "secret".to_owned(),
        "http://113.46.219.251:8080/v1".to_owned(),
        "GLM-5.3-Flash".to_owned(),
    )
    .unwrap();

    assert_eq!(
        config.chat_completions_url,
        "http://113.46.219.251:8080/v1/chat/completions"
    );
    assert_eq!(config.model, "GLM-5.3-Flash");
}

#[test]
fn preserves_complete_chat_completions_url() {
    let config = LocalOpenAIConfig::from_values(
        "secret".to_owned(),
        "https://example.com/v1/chat/completions".to_owned(),
        "model".to_owned(),
    )
    .unwrap();

    assert_eq!(
        config.chat_completions_url,
        "https://example.com/v1/chat/completions"
    );
}

#[test]
fn rejects_non_http_schemes_and_blank_fields() {
    assert!(
        LocalOpenAIConfig::from_values(
            "secret".to_owned(),
            "file:///tmp/model".to_owned(),
            "model".to_owned(),
        )
        .is_err()
    );
    assert!(
        LocalOpenAIConfig::from_values(
            " ".to_owned(),
            "https://example.com/v1".to_owned(),
            "model".to_owned(),
        )
        .is_err()
    );
}

#[test]
fn extracts_user_queries_from_batched_input() {
    let input = api::request::Input {
        r#type: Some(api::request::input::Type::UserInputs(
            api::request::input::UserInputs {
                inputs: vec![api::request::input::user_inputs::UserInput {
                    input: Some(
                        api::request::input::user_inputs::user_input::Input::UserQuery(
                            api::request::input::UserQuery {
                                query: "hello".to_owned(),
                                ..Default::default()
                            },
                        ),
                    ),
                }],
            },
        )),
        ..Default::default()
    };

    assert_eq!(queries_from_input(&input), ["hello"]);
}

#[test]
fn response_contains_init_task_messages_and_done() {
    let mut params = RequestParams::new_for_test();
    params.input = Vec::new();
    let events = response_events(
        params,
        vec!["hello".to_owned()],
        "world".to_owned(),
        "model".to_owned(),
    );

    assert!(matches!(
        events[0].r#type,
        Some(api::response_event::Type::Init(_))
    ));
    let Some(api::response_event::Type::ClientActions(actions)) = &events[1].r#type else {
        panic!("expected client actions");
    };
    assert_eq!(actions.actions.len(), 2);
    assert!(matches!(
        events[2].r#type,
        Some(api::response_event::Type::Finished(_))
    ));
}

#[test]
fn parses_generated_commands_from_plain_and_fenced_json() {
    let expected = vec![LocalGeneratedCommand {
        command: "git status".to_owned(),
        description: "Show repository status".to_owned(),
    }];
    let json = r#"{"commands":[{"command":"git status","description":"Show repository status"}]}"#;
    let fenced_json = format!("```json\n{json}\n```");

    assert_eq!(parse_generated_commands(json).unwrap(), expected);
    assert_eq!(parse_generated_commands(&fenced_json).unwrap(), expected);
}

#[test]
fn parses_generated_command_metadata() {
    let metadata = parse_json_response::<LocalGeneratedCommandMetadata>(
        r#"{
            "command": "curl {{url}}",
            "title": "Fetch URL",
            "description": "Download a URL",
            "arguments": [{
                "name": "url",
                "description": "URL to fetch",
                "default_value": "https://example.com"
            }]
        }"#,
    )
    .unwrap();

    assert_eq!(metadata.command, "curl {{url}}");
    assert_eq!(metadata.title, "Fetch URL");
    assert_eq!(metadata.arguments.len(), 1);
    assert_eq!(metadata.arguments[0].name, "url");
}
