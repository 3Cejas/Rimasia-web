(() => {
  const MAX_VIEW_SCALE = 7.5;

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function parseJsonScript(id, fallback) {
    const node = document.getElementById(id);
    if (!node) {
      return fallback;
    }

    try {
      return JSON.parse(node.textContent || '');
    } catch {
      return fallback;
    }
  }

  function approach(value, target, rate, deltaMs) {
    const blend = 1 - Math.exp(-(rate * deltaMs) / 1000);
    return value + (target - value) * blend;
  }

  function initBackgroundSky() {
    const canvas = document.getElementById('embedding-sky');
    const dock = document.querySelector('.search-dock');
    const payload = parseJsonScript('embedding-graph', { nodes: [], edges: [] });
    let highlightedWords = new Set(parseJsonScript('embedding-highlights', []));
    let queryPayload = parseJsonScript('query-constellation-data', { nodes: [], edges: [] });

    if (!canvas || !payload.nodes || !payload.edges) {
      return;
    }

    const ctx = canvas.getContext('2d');
    const baseNodes = (payload.nodes || []).map((node, index) => ({
      ...node,
      phase: index * 0.63,
      drift: 0.003 + (index % 5) * 0.0007,
      pulse: 0.8 + (index % 7) * 0.07,
    }));
    function buildQueryNodes(nodes) {
      const groupOffsets = { consonant: 0.08, assonant: 0.3 };
      const groupCounters = { consonant: 0, assonant: 0 };
      return (nodes || []).map((node, index) => {
        const order = groupCounters[node.group] || 0;
        groupCounters[node.group] = order + 1;
        return {
          ...node,
          phase: index * 0.77,
          drift: 0.0032,
          pulse: 0.82 + (index % 6) * 0.08,
          revealDelay: (groupOffsets[node.group] || 0.08) + order * 0.07,
        };
      });
    }
    const edges = payload.edges || [];
    let queryNodes = [];
    let queryEdges = [];
    const denseBackground = baseNodes.length > 1500;
    let width = 0;
    let height = 0;
    let viewScale = 1;
    let panX = 0;
    let panY = 0;
    let draggingPointerId = null;
    let dragStartClientX = 0;
    let dragStartClientY = 0;
    let dragStartPanX = 0;
    let dragStartPanY = 0;
    let dragMoved = false;
    let sceneCenterX = 0;
    let sceneCenterY = 0;
    let dockInset = 0;
    const baseLabelState = new Map();
    let previousFrameAt = performance.now();

    function updateViewportCenter() {
      if (!dock || width <= 900) {
        dockInset = 0;
      } else {
        const dockRect = dock.getBoundingClientRect();
        dockInset = clamp(dockRect.right + 32, 0, width * 0.42);
      }

      sceneCenterX = dockInset + (width - dockInset) * 0.5;
      sceneCenterY = height * 0.5;
    }

    function panBounds() {
      const baseX = Math.max(96, width * Math.abs(viewScale - 1) * 0.62);
      const baseY = Math.max(72, height * Math.abs(viewScale - 1) * 0.56);
      const dockTravel = dockInset * 0.95;
      return {
        minX: -(baseX + dockTravel),
        maxX: baseX + dockTravel * 0.55,
        minY: -baseY,
        maxY: baseY,
      };
    }

    function clampPan() {
      const bounds = panBounds();
      panX = clamp(panX, bounds.minX, bounds.maxX);
      panY = clamp(panY, bounds.minY, bounds.maxY);
    }

    function screenToWorld(screenX, screenY) {
      return {
        x: (screenX - sceneCenterX - panX) / viewScale + sceneCenterX,
        y: (screenY - sceneCenterY - panY) / viewScale + sceneCenterY,
      };
    }

    function applyView(worldX, worldY) {
      return {
        x: (worldX - sceneCenterX) * viewScale + sceneCenterX + panX,
        y: (worldY - sceneCenterY) * viewScale + sceneCenterY + panY,
      };
    }

    function resize() {
      const ratio = window.devicePixelRatio || 1;
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = Math.floor(width * ratio);
      canvas.height = Math.floor(height * ratio);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      updateViewportCenter();
      clampPan();
    }

    function syncDockViewport() {
      updateViewportCenter();
      clampPan();
    }

    function pointFor(node, time, motionScale = 1) {
      const swayX = Math.sin(time * 0.00011 + node.phase) * node.drift * width * motionScale;
      const swayY = Math.cos(time * 0.00008 + node.phase * 1.3) * node.drift * height * motionScale;
      const worldX = node.x * width + swayX;
      const worldY = node.y * height + swayY;
      return applyView(worldX, worldY);
    }

    function drawGlow(x, y, radius, color, alpha) {
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius * 4.8);
      gradient.addColorStop(0, `rgba(${color}, ${alpha})`);
      gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(x, y, radius * 4.8, 0, Math.PI * 2);
      ctx.fill();
    }

    function easeOutCubic(progress) {
      return 1 - ((1 - progress) ** 3);
    }

    function stagedProgress(globalProgress, delay, span) {
      return clamp((globalProgress - delay) / span, 0, 1);
    }

    function queryNodePalette(node) {
      if (node.group === 'consonant') {
        return {
          core: node.is_fallback ? '246, 193, 111' : '172, 164, 255',
          glow: node.is_fallback ? '246, 193, 111' : '172, 164, 255',
          label: node.is_fallback ? 'rgba(255, 239, 194, 0.98)' : 'rgba(241, 236, 255, 0.99)',
          labelGlow: node.is_fallback ? '246, 193, 111' : '172, 164, 255',
        };
      }

      return {
        core: node.is_fallback ? '246, 193, 111' : '125, 230, 255',
        glow: node.is_fallback ? '246, 193, 111' : '125, 230, 255',
        label: node.is_fallback ? 'rgba(255, 239, 194, 0.98)' : 'rgba(228, 252, 255, 0.99)',
        labelGlow: node.is_fallback ? '246, 193, 111' : '125, 230, 255',
      };
    }

    function queryEdgeColor(edge, source, target) {
      const hasAsonant = source?.group === 'assonant' || target?.group === 'assonant';
      const hasConsonant = source?.group === 'consonant' || target?.group === 'consonant';
      if (hasConsonant && !hasAsonant) {
        return `rgba(172, 164, 255, ${0.18 + edge.strength * 0.46})`;
      }
      if (hasAsonant) {
        return `rgba(125, 230, 255, ${0.18 + edge.strength * 0.46})`;
      }
      return `rgba(246, 193, 111, ${0.16 + edge.strength * 0.38})`;
    }

    function drawLabel(x, y, text, fillStyle, font, zoomVisualScale, alpha = 1) {
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.fillStyle = fillStyle;
      ctx.font = font;
      ctx.fillText(text, x + 11 * zoomVisualScale, y - 10 * zoomVisualScale);
      ctx.restore();
    }

    function drawQueryLabel(x, y, text, palette, font, zoomVisualScale, alpha) {
      const labelX = x + 12 * zoomVisualScale;
      const labelY = y - 11 * zoomVisualScale;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.font = font;
      ctx.lineJoin = 'round';
      ctx.strokeStyle = 'rgba(3, 6, 18, 0.92)';
      ctx.lineWidth = 4.4;
      ctx.shadowBlur = 18;
      ctx.shadowColor = `rgba(${palette.labelGlow}, 0.34)`;
      ctx.strokeText(text, labelX, labelY);
      ctx.fillStyle = palette.label;
      ctx.fillText(text, labelX, labelY);
      ctx.restore();
    }

    function isVisible(x, y, margin = 64) {
      return x >= -margin && x <= width + margin && y >= -margin && y <= height + margin;
    }

    function pointerPosition(event) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
      };
    }

    function drawBaseLabels(nodes, zoomVisualScale, deltaMs) {
      const visibleNodes = nodes.filter((node) => isVisible(node.x, node.y) && node.word);
      const targetLabels = new Map();

      if (visibleNodes.length) {
        const zoomFactor = clamp(viewScale, 0.7, MAX_VIEW_SCALE);
        const maxLabels = zoomFactor >= 2
          ? visibleNodes.length
          : Math.round(clamp(150 * zoomFactor * zoomFactor, 120, 980));
        const cellSize = clamp(92 / zoomFactor, 26, 92);
        const occupied = new Set();
        let labelsDrawn = 0;

        const labelCandidates = visibleNodes.sort((left, right) => {
          if (left.highlight !== right.highlight) {
            return left.highlight ? -1 : 1;
          }
          return right.size - left.size;
        });

        for (const node of labelCandidates) {
          if (!node.highlight && labelsDrawn >= maxLabels) {
            break;
          }

          const labelFontSize = node.highlight ? 12 : 10;
          const scaledFontSize = Math.round(
            labelFontSize * clamp(viewScale ** 0.32, 0.88, 1.5),
          );
          const labelWidth = 14 + node.word.length * scaledFontSize * 0.56;
          const originX = node.x + 11 * zoomVisualScale;
          const originY = node.y - 10 * zoomVisualScale;
          const gridX = Math.floor(originX / cellSize);
          const gridY = Math.floor(originY / cellSize);
          const spanX = Math.max(1, Math.ceil(labelWidth / cellSize));
          const spanY = 2;
          let blocked = false;

          if (!node.highlight) {
            for (let offsetY = 0; offsetY < spanY; offsetY += 1) {
              for (let offsetX = 0; offsetX < spanX; offsetX += 1) {
                if (occupied.has(`${gridX + offsetX}:${gridY + offsetY}`)) {
                  blocked = true;
                  break;
                }
              }
              if (blocked) {
                break;
              }
            }
          }

          if (blocked) {
            continue;
          }

          for (let offsetY = 0; offsetY < spanY; offsetY += 1) {
            for (let offsetX = 0; offsetX < spanX; offsetX += 1) {
              occupied.add(`${gridX + offsetX}:${gridY + offsetY}`);
            }
          }

          targetLabels.set(node.word, {
            x: node.x,
            y: node.y,
            word: node.word,
            font: `${node.highlight ? '600' : '500'} ${scaledFontSize}px Trebuchet MS`,
            fillStyle: node.highlight ? 'rgba(125, 230, 255, 0.92)' : 'rgba(166, 185, 225, 0.78)',
            highlight: node.highlight,
          });
          labelsDrawn += 1;
        }
      }

      for (const [word, label] of targetLabels) {
        const existing = baseLabelState.get(word);
        if (existing) {
          existing.x = label.x;
          existing.y = label.y;
          existing.font = label.font;
          existing.fillStyle = label.fillStyle;
          existing.highlight = label.highlight;
          existing.alpha = approach(existing.alpha, 1, 11.5, deltaMs);
        } else {
          baseLabelState.set(word, {
            ...label,
            alpha: approach(0, 1, 11.5, deltaMs),
          });
        }
      }

      for (const [word, label] of baseLabelState) {
        if (targetLabels.has(word)) {
          continue;
        }

        label.alpha = approach(label.alpha, 0, 7.8, deltaMs);
        if (label.alpha <= 0.035) {
          baseLabelState.delete(word);
        }
      }

      const drawableLabels = Array.from(baseLabelState.values())
        .filter((label) => label.alpha > 0.035 && isVisible(label.x, label.y, 120))
        .sort((left, right) => {
          if (left.highlight !== right.highlight) {
            return left.highlight ? 1 : -1;
          }
          return left.alpha - right.alpha;
        });

      for (const label of drawableLabels) {
        drawLabel(
          label.x,
          label.y,
          label.word,
          label.fillStyle,
          label.font,
          zoomVisualScale,
          label.alpha,
        );
      }
    }

    let animationStartedAt = performance.now();

    function applySearchState(nextQueryPayload, nextHighlightWords = []) {
      highlightedWords = new Set(nextHighlightWords || []);
      queryPayload = nextQueryPayload || { nodes: [], edges: [] };
      queryNodes = buildQueryNodes(queryPayload.nodes || []);
      queryEdges = queryPayload.edges || [];
      animationStartedAt = performance.now();
    }

    queryNodes = buildQueryNodes(queryPayload.nodes || []);
    queryEdges = queryPayload.edges || [];

    window.RimasiaConstellation = {
      applySearchState,
    };

    function draw(time) {
      const deltaMs = Math.min(64, Math.max(16, time - previousFrameAt));
      previousFrameAt = time;
      ctx.clearRect(0, 0, width, height);

      const sky = ctx.createLinearGradient(0, 0, 0, height);
      sky.addColorStop(0, 'rgba(3, 5, 16, 0.35)');
      sky.addColorStop(1, 'rgba(2, 3, 10, 0.92)');
      ctx.fillStyle = sky;
      ctx.fillRect(0, 0, width, height);

      const projected = baseNodes.map((node) => ({
        ...node,
        ...pointFor(node, time),
        highlight: highlightedWords.has(node.word),
      }));
      const projectedQuery = queryNodes.map((node) => ({
        ...node,
        ...pointFor(node, time, 0.35),
      }));
      const queryLookup = new Map(projectedQuery.map((node) => [node.id, node]));
      const zoomVisualScale = clamp(viewScale ** 0.82, 0.68, 2.8);

      ctx.lineWidth = 1;
      for (const edge of edges) {
        const source = projected[edge.source];
        const target = projected[edge.target];
        if (!source || !target) {
          continue;
        }

        const highlighted = source.highlight || target.highlight;
        ctx.strokeStyle = highlighted
          ? `rgba(125, 230, 255, ${0.1 + edge.strength * 0.35})`
          : `rgba(110, 143, 255, ${0.04 + edge.strength * 0.18})`;
        ctx.beginPath();
        ctx.moveTo(source.x, source.y);
        ctx.lineTo(target.x, target.y);
        ctx.stroke();
      }

      for (const node of projected) {
        const pulse = 0.75 + Math.sin(time * 0.0014 * node.pulse + node.phase) * 0.18;
        const radius = Math.max(1.2, node.size * pulse * zoomVisualScale);
        const color = node.highlight ? '125, 230, 255' : '234, 241, 255';
        const alpha = node.highlight ? 0.26 : 0.38;
        const showGlow = node.highlight || !denseBackground || node.size >= 1.95;
        if (showGlow) {
          drawGlow(node.x, node.y, radius, color, alpha);
        }

        ctx.fillStyle = node.highlight
          ? 'rgba(125, 230, 255, 0.97)'
          : denseBackground
            ? 'rgba(252, 253, 255, 0.9)'
            : 'rgba(252, 253, 255, 1)';
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
        ctx.fill();

        if (!node.highlight) {
          ctx.fillStyle = 'rgba(255, 255, 255, 1)';
          ctx.beginPath();
          ctx.arc(node.x, node.y, Math.max(1.15, radius * 0.4), 0, Math.PI * 2);
          ctx.fill();
        }
      }

      drawBaseLabels(projected, zoomVisualScale, deltaMs);

      const queryGlobalProgress = clamp((time - animationStartedAt) / 1700, 0, 1);

      for (const edge of queryEdges) {
        const source = queryLookup.get(edge.source);
        const target = queryLookup.get(edge.target);
        if (!source || !target) {
          continue;
        }

        const sourceProgress = easeOutCubic(stagedProgress(queryGlobalProgress, source.revealDelay, 0.16));
        const targetProgress = easeOutCubic(stagedProgress(queryGlobalProgress, target.revealDelay, 0.16));
        const edgeProgress = Math.min(sourceProgress, targetProgress);
        if (edgeProgress <= 0) {
          continue;
        }

        ctx.strokeStyle = queryEdgeColor(edge, source, target);
        ctx.globalAlpha = edgeProgress;
        ctx.lineWidth = 1.4 + edge.strength * 2.2;
        ctx.beginPath();
        ctx.moveTo(source.x, source.y);
        ctx.lineTo(
          source.x + (target.x - source.x) * edgeProgress,
          source.y + (target.y - source.y) * edgeProgress,
        );
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      for (const node of projectedQuery) {
        const nodeProgress = easeOutCubic(stagedProgress(queryGlobalProgress, node.revealDelay, 0.18));
        if (nodeProgress <= 0) {
          continue;
        }

        const palette = queryNodePalette(node);
        const pulse = 0.84 + Math.sin(time * 0.0013 * node.pulse + node.phase) * 0.14;
        const radius = Math.max(7, node.size * 8.8 * pulse * zoomVisualScale) * nodeProgress;

        drawGlow(node.x, node.y, radius, palette.glow, 0.22 * nodeProgress);

        ctx.globalAlpha = nodeProgress;
        ctx.fillStyle = `rgba(${palette.core}, 0.96)`;
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = 'rgba(255, 255, 255, 0.78)';
        ctx.beginPath();
        ctx.arc(node.x, node.y, Math.max(1.8, radius * 0.22), 0, Math.PI * 2);
        ctx.fill();

        const font = `700 ${Math.round(15 * clamp(viewScale ** 0.36, 0.94, 1.9))}px Trebuchet MS`;
        drawQueryLabel(node.x, node.y, node.word, palette, font, zoomVisualScale, nodeProgress);
        ctx.globalAlpha = 1;
      }

      requestAnimationFrame(draw);
    }

    window.addEventListener('resize', resize);
    window.addEventListener('rimasia-dock-resize', syncDockViewport);
    if (dock && 'ResizeObserver' in window) {
      const dockObserver = new ResizeObserver(() => {
        syncDockViewport();
      });
      dockObserver.observe(dock);
    }
    canvas.addEventListener('wheel', (event) => {
      event.preventDefault();
      const pointer = pointerPosition(event);

      const nextScale = clamp(
        viewScale * (event.deltaY < 0 ? 1.12 : 1 / 1.12),
        0.7,
        MAX_VIEW_SCALE,
      );
      if (nextScale === viewScale) {
        return;
      }

      const worldPoint = screenToWorld(pointer.x, pointer.y);
      viewScale = nextScale;
      const projectedPoint = applyView(worldPoint.x, worldPoint.y);
      panX += pointer.x - projectedPoint.x;
      panY += pointer.y - projectedPoint.y;
      clampPan();
    }, { passive: false });
    canvas.addEventListener('pointerdown', (event) => {
      draggingPointerId = event.pointerId;
      dragStartClientX = event.clientX;
      dragStartClientY = event.clientY;
      dragStartPanX = panX;
      dragStartPanY = panY;
      dragMoved = false;
      canvas.classList.add('is-dragging');
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener('pointermove', (event) => {
      if (event.pointerId !== draggingPointerId) {
        return;
      }

      if (
        Math.abs(event.clientX - dragStartClientX) > 4
        || Math.abs(event.clientY - dragStartClientY) > 4
      ) {
        dragMoved = true;
      }
      panX = dragStartPanX + (event.clientX - dragStartClientX);
      panY = dragStartPanY + (event.clientY - dragStartClientY);
      clampPan();
    });

    function endDrag(event) {
      if (event.pointerId !== draggingPointerId) {
        return;
      }

      draggingPointerId = null;
      canvas.classList.remove('is-dragging');
      if (canvas.hasPointerCapture(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
    }

    canvas.addEventListener('pointerup', endDrag);
    canvas.addEventListener('pointercancel', endDrag);

    resize();
    requestAnimationFrame(draw);
  }

  initBackgroundSky();
})();
