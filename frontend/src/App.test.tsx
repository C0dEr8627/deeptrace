import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import App from "./App";

describe("DeepTrace upload shell", () => {
  it("renders the image-first upload interface", () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: /see beyond the surface/i })).toBeTruthy();
    expect(screen.getByLabelText(/choose image/i)).toBeTruthy();
  });

  it("rejects unsupported media types", () => {
    render(<App />);
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    const unsupported = new File(["not an image"], "sample.gif", { type: "image/gif" });
    fireEvent.change(input, { target: { files: [unsupported] } });
    expect(screen.getByRole("alert").textContent).toMatch(/choose a jpeg, png, or webp image/i);
  });
});
