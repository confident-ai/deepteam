
const BUCKETS = {
  deepteamDocs: "https://deepteam-docs.s3.amazonaws.com",
  deepteamDocsRegion: "https://deepteam-docs.s3.us-east-1.amazonaws.com",
  confidentDocs: "https://confident-docs.s3.us-east-1.amazonaws.com",
  confidentBucket: "https://confident-bucket.s3.us-east-1.amazonaws.com",
};

export const ASSETS = {
  // Image assets
  confidentRedTeamingChooseFramework: `${BUCKETS.confidentDocs}/red-teaming:fameworks:choose-framework.png`,
  confidentRedTeamingCreateFramework: `${BUCKETS.confidentDocs}/red-teaming:frameworks:create-framework.png`,
  confidentRedTeamingCustomizeRiskCategory: `${BUCKETS.confidentDocs}/red-teaming:frameworks:customize-risk-category.png`,
  confidentRedTeamingRiskAssessment: `${BUCKETS.confidentDocs}/red-teaming:quick-start:risk-assessment.png`,
  confidentRedTeamingRunRiskAssessment: `${BUCKETS.confidentDocs}/red-teaming:quick-start:run-risk-assessment.png`,
  confidentRedTeamingRiskAssessmentTestCases: `${BUCKETS.confidentDocs}/red-teaming:risk-profile:risk-assessment-test-cases.png`,
  confidentRedTeamingScheduledFramework: `${BUCKETS.confidentDocs}/red-teaming:scheduled-red-team-framework.png`,
  confidentSettingsProjectAiConnection: `${BUCKETS.confidentDocs}/settings:project:ai-connection.png`,
  // Video assets
  redTeamingOwaspTop10LlmPlatformVideo: `${BUCKETS.deepteamDocs}/red-teaming:owasp-top-10-llm.mp4`,
};
