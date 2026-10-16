const BUCKETS = {
  deepteamDocs: "https://deepteam-docs.s3.amazonaws.com",
};

// Videos stay on S3. Images live in the media library (docs/media/) and are
// referenced by id through <ImageDisplayer id="..." />.
export const ASSETS = {
  // ---- Platform videos ----
  redTeamingOwaspTop10LlmPlatformVideo: `${BUCKETS.deepteamDocs}/red-teaming:owasp-top-10-llm.mp4`,
};
