const BUCKETS = {
  deepteamDocs: "https://deepteam-docs.s3.amazonaws.com",
  deepteamDocsRegion: "https://deepteam-docs.s3.us-east-1.amazonaws.com",
  confidentDocs: "https://confident-docs.s3.us-east-1.amazonaws.com",
};

export const ASSETS = {
  // ---- Concept diagrams ----
  redTeamingIntro: `${BUCKETS.deepteamDocs}/red-teaming-intro.png`,
  redTeamingWorkflow: `${BUCKETS.deepteamDocs}/red-teaming-workflow.svg`,
  modelVsSystemWeakness: `${BUCKETS.deepteamDocs}/model-vs-system-weakness.svg`,
  redTeamingProcess: `${BUCKETS.deepteamDocsRegion}/docs:red-teaming-process.png`,
  redTeamingTestCase: `${BUCKETS.deepteamDocsRegion}/docs:red-teaming-test-case.png`,
  redTeamingTestCaseTurns: `${BUCKETS.deepteamDocsRegion}/docs:red-teaming-test-case-turns.png`,
  redTeamingRiskAssessment: `${BUCKETS.deepteamDocsRegion}/docs:red-teaming-risk-assessment.png`,

  // ---- Vulnerability examples ----
  biasExample: `${BUCKETS.deepteamDocs}/bias-example.png`,
  biasGemini: `${BUCKETS.deepteamDocs}/bias-gemini.png`,
  piiLeakageExample: `${BUCKETS.deepteamDocs}/pii-leakage-example.png`,
  piiLeakageGpt2: `${BUCKETS.deepteamDocs}/pii-leakage-gpt2.png`,
  promptInjectionExample: `${BUCKETS.deepteamDocs}/prompt-injection-example.png`,

  // ---- Jailbreaking concept images ----
  jailbreakingExample: `${BUCKETS.deepteamDocs}/jailbreaking-example.png`,
  jailbreakingDanExample: `${BUCKETS.deepteamDocs}/jailbreaking-dan-example.png`,
  manyShotJailbreakingExample: `${BUCKETS.deepteamDocs}/many-shot-jailbreaking-example.png`,

  // ---- Multi-turn attack diagrams ----
  attackBadLikertJudge: `${BUCKETS.deepteamDocsRegion}/attacks:bad-likert-judge.png`,
  attackBadLikertJudgeExample: `${BUCKETS.deepteamDocsRegion}/attacks:bad-likert-judge-example.png`,
  attackCrescendoJailbreaking: `${BUCKETS.deepteamDocsRegion}/attacks:crescendo-jailbreaking.png`,
  attackCrescendoJailbreakingExample: `${BUCKETS.deepteamDocsRegion}/attacks:crescendo-jailbreaking-example.png`,
  attackLinearJailbreakingIntro: `${BUCKETS.deepteamDocsRegion}/attacks:linear-jailbreaking-intro.png`,
  attackLinearJailbreakingExample: `${BUCKETS.deepteamDocsRegion}/attacks:linear-jailbreaking-example.png`,
  attackSequentialJailbreaking: `${BUCKETS.deepteamDocsRegion}/attacks:sequential-jailbreaking.png`,
  attackSequentialJailbreakingExample: `${BUCKETS.deepteamDocsRegion}/attacks:sequential-jailbreaking-example.png`,
  attackTreeJailbreakingIntro: `${BUCKETS.deepteamDocsRegion}/attacks:tree-jailbreaking-intro.png`,
  attackTreeJailbreaking: `${BUCKETS.deepteamDocsRegion}/attacks:tree-jailbreaking.png`,
  attackTreeJailbreakingExample: `${BUCKETS.deepteamDocsRegion}/attacks:tree-jailbreaking-example.png`,

  // ---- Guide customization diagrams ----
  guideSingleTurnCustomization: `${BUCKETS.deepteamDocsRegion}/guides:single-turn-customization.png`,
  guideMultiTurnCustomization: `${BUCKETS.deepteamDocsRegion}/guides:multi-turn-customization.png`,

  // ---- Confident AI platform screenshots ----
  confidentRedTeamingChooseFramework: `${BUCKETS.confidentDocs}/red-teaming:fameworks:choose-framework.png`,
  confidentRedTeamingCreateFramework: `${BUCKETS.confidentDocs}/red-teaming:frameworks:create-framework.png`,
  confidentRedTeamingCustomizeRiskCategory: `${BUCKETS.confidentDocs}/red-teaming:frameworks:customize-risk-category.png`,
  confidentRedTeamingRiskAssessment: `${BUCKETS.confidentDocs}/red-teaming:quick-start:risk-assessment.png`,
  confidentRedTeamingRunRiskAssessment: `${BUCKETS.confidentDocs}/red-teaming:quick-start:run-risk-assessment.png`,
  confidentRedTeamingRiskAssessmentTestCases: `${BUCKETS.confidentDocs}/red-teaming:risk-profile:risk-assessment-test-cases.png`,
  confidentRedTeamingScheduledFramework: `${BUCKETS.confidentDocs}/red-teaming:scheduled-red-team-framework.png`,
  confidentSettingsProjectAiConnection: `${BUCKETS.confidentDocs}/settings:project:ai-connection.png`,

  // ---- Platform videos ----
  redTeamingOwaspTop10LlmPlatformVideo: `${BUCKETS.deepteamDocs}/red-teaming:owasp-top-10-llm.mp4`,
};
