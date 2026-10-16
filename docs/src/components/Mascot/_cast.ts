import type { Character } from ".";

export const CAST = {
  classic: {
    color: "color-mix(in srgb, #de0064 45%, white)",
    speed: 5,
    hover: "wave",
  },
} satisfies Record<string, Character>;

export const CLICKS = ["jump", "dance", "shake", "cry", "freeze"] as const;
