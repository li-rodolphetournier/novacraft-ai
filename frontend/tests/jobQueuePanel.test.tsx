import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { JobQueuePanel } from "@/components/generator/JobQueuePanel";
import type { Job } from "@/types/generator";

const baseHandlers = {
  onSelectJob: vi.fn(),
  onPause: vi.fn(),
  onResume: vi.fn(),
  onStart: vi.fn(),
  onCancel: vi.fn(),
  onDelete: vi.fn(),
  onClearCompleted: vi.fn(),
};

afterEach(() => {
  vi.clearAllMocks();
});

const makeJobs = (): Job[] => [
  {
    id: "job-running",
    type: "image",
    status: "running",
    progress: { current: 1, total: 4, message: "1/4" },
    metadata: { prompt_preview: "Image hyper réaliste" },
    created_at: Date.now() / 1000,
  },
  {
    id: "job-pending",
    type: "video",
    status: "pending",
    progress: { current: 0, total: 8 },
    metadata: { prompt_preview: "Vidéo stylisée" },
    created_at: Date.now() / 1000,
  },
];

describe("JobQueuePanel", () => {
  it("affiche les jobs avec icônes et statuts", () => {
    render(
      <JobQueuePanel
        jobs={makeJobs()}
        selectedJobId={null}
        hasRunningJob
        jobsError={null}
        {...baseHandlers}
      />,
    );

    expect(screen.getByText("🖼️")).toBeInTheDocument();
    expect(screen.getByText("🎞️")).toBeInTheDocument();
    expect(screen.getByText(/En cours/i)).toBeInTheDocument();
    expect(screen.getByText(/En file/i)).toBeInTheDocument();
  });

  it("désactive le bouton Reprendre si un job est déjà en cours", () => {
    render(
      <JobQueuePanel
        jobs={makeJobs()}
        selectedJobId={null}
        hasRunningJob
        jobsError={null}
        {...baseHandlers}
      />,
    );

    const startButton = screen.getByRole("button", { name: /Reprendre/i });
    expect(startButton).toBeDisabled();
  });

  it("active le bouton Reprendre quand aucun job ne tourne", () => {
    render(
      <JobQueuePanel
        jobs={makeJobs()}
        selectedJobId={null}
        hasRunningJob={false}
        jobsError={null}
        {...baseHandlers}
      />,
    );

    const startButton = screen.getByRole("button", { name: /Reprendre/i });
    expect(startButton).not.toBeDisabled();
  });

  it("affiche un bouton pour supprimer les jobs annulables", () => {
    render(
      <JobQueuePanel
        jobs={makeJobs()}
        selectedJobId={null}
        hasRunningJob={false}
        jobsError={null}
        {...baseHandlers}
      />,
    );

    const deleteButtons = screen.getAllByRole("button", { name: /retirer le job/i });
    expect(deleteButtons.length).toBeGreaterThan(0);
  });

  it("déclenche onDelete pour les jobs non actifs", () => {
    const jobs: Job[] = [
      {
        id: "job-completed",
        type: "image",
        status: "completed",
        progress: { current: 4, total: 4 },
        metadata: { prompt_preview: "Terminé" },
        created_at: Date.now() / 1000,
      },
    ];

    render(
      <JobQueuePanel
        jobs={jobs}
        selectedJobId={null}
        hasRunningJob={false}
        jobsError={null}
        {...baseHandlers}
      />,
    );

    const deleteButton = screen.getByRole("button", { name: /retirer le job/i });
    fireEvent.click(deleteButton);
    expect(baseHandlers.onDelete).toHaveBeenCalledWith("job-completed");
  });
});

