import { Component, type ErrorInfo, type ReactNode } from 'react';

export interface WidgetErrorBoundaryProps {
  readonly children: ReactNode;
  readonly resetKey?: string;
  readonly onReset?: () => void;
  readonly onFallbackToGeneric?: () => void;
}

interface WidgetErrorBoundaryState {
  readonly hasError: boolean;
  readonly error: Error | null;
  readonly errorInfo: ErrorInfo | null;
}

/**
 * Robust React Error Boundary wrapping domain widget viewports.
 *
 * Catches and isolates unhandled runtime rendering exceptions (e.g. WebGL context loss,
 * malformed GLSL shaders, or invalid AST structures) without crashing the parent IDE.
 */
export class WidgetErrorBoundary extends Component<
  WidgetErrorBoundaryProps,
  WidgetErrorBoundaryState
> {
  public override state: WidgetErrorBoundaryState = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  public static getDerivedStateFromError(error: Error): Partial<WidgetErrorBoundaryState> {
    return { hasError: true, error };
  }

  public override componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error('[WidgetErrorBoundary] Caught unhandled widget exception:', {
      errorName: error.name,
      errorMessage: error.message,
      componentStack: errorInfo.componentStack,
      resetKey: this.props.resetKey,
      timestamp: new Date().toISOString(),
    });
    this.setState({ errorInfo });
  }

  public override componentDidUpdate(prevProps: WidgetErrorBoundaryProps): void {
    if (this.state.hasError && prevProps.resetKey !== this.props.resetKey) {
      // Clear error state when the reset key changes without triggering onReset callback loop
      this.setState({ hasError: false, error: null, errorInfo: null });
    }
  }

  private handleReset = (): void => {
    this.setState({ hasError: false, error: null, errorInfo: null });
    this.props.onReset?.();
  };

  public override render(): ReactNode {
    if (this.state.hasError) {
      const { error } = this.state;
      return (
        <div
          className="widget-stage widget-error-boundary-container"
          data-testid="widget-error-boundary-fallback"
        >
          <div className="stage-card widget-error-card">
            <div className="stage-icon" aria-hidden="true">⚠️</div>
            <h2 className="stage-title" style={{ color: 'var(--accent-red)' }}>
              Widget Runtime Exception Isolated
            </h2>
            <p className="stage-description">
              The active domain visualizer encountered an unexpected error during rendering. The surrounding IDE remains operational.
            </p>

            <div className="error-message-box">
              <span className="error-name">{error?.name || 'Error'}:</span>{' '}
              <span className="error-msg">{error?.message || 'Unknown runtime exception'}</span>
            </div>

            <div className="widget-error-actions" style={{ marginTop: '16px', display: 'flex', gap: '8px' }}>
              <button
                type="button"
                className="btn btn-primary"
                onClick={this.handleReset}
                data-testid="widget-error-retry-btn"
              >
                🔄 Retry Render
              </button>
              {this.props.onFallbackToGeneric && (
                <button
                  type="button"
                  className="btn"
                  onClick={this.props.onFallbackToGeneric}
                  data-testid="widget-error-fallback-generic-btn"
                >
                  🌳 View Generic AST
                </button>
              )}
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
