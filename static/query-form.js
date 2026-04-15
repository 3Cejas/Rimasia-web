(() => {
  const editor = document.getElementById('concept-editor');
  const chipsContainer = document.getElementById('concept-chips');
  const input = document.getElementById('temas-input');
  const hiddenInput = document.getElementById('temas');
  const targetInput = document.getElementById('palabra');
  const messageRegion = document.getElementById('form-message-region');
  const queryScript = document.getElementById('query-constellation-data');
  const highlightScript = document.getElementById('embedding-highlights');
  const resultPageInput = document.getElementById('pagina');
  const exploreControls = document.getElementById('explore-controls');
  const exploreMoreButton = document.getElementById('explore-more-button');
  const searchDock = document.getElementById('search-dock');
  const dockToggle = document.getElementById('dock-toggle');
  const dockContent = document.getElementById('search-dock-content');

  if (!editor || !chipsContainer || !input || !hiddenInput) {
    return;
  }

  const DOCK_STORAGE_KEY = 'rimasia:dock-collapsed';

  let initialQueryPayload = { nodes: [], edges: [] };
  if (queryScript) {
    try {
      initialQueryPayload = JSON.parse(queryScript.textContent || '');
    } catch {
      initialQueryPayload = { nodes: [], edges: [] };
    }
  }

  const initialValues = hiddenInput.value
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean);

  let concepts = [];
  let isLoading = false;
  let currentPage = Number(resultPageInput instanceof HTMLInputElement ? resultPageInput.value : 0) || 0;
  let currentPageCount = Number(exploreControls?.dataset.pageCount || 1) || 1;
  let currentConsonantTotal = Number(exploreControls?.dataset.consonantTotal || 0) || 0;
  let currentAssonantTotal = Number(exploreControls?.dataset.assonantTotal || 0) || 0;
  let dockCollapsedPreference = false;
  let loadingDisplayProgress = 0;
  let loadingTargetProgress = 0;
  let loadingRafId = 0;
  let lastLoadingFrame = performance.now();

  try {
    dockCollapsedPreference = window.localStorage.getItem(DOCK_STORAGE_KEY) === '1';
  } catch {
    dockCollapsedPreference = false;
  }

  function normalize(value) {
    return value.trim().toLowerCase();
  }

  function toDisplayCapitalized(value) {
    const trimmedValue = value.trim();
    if (!trimmedValue) {
      return '';
    }

    const [firstCharacter, ...rest] = Array.from(trimmedValue);
    return `${firstCharacter.toLocaleUpperCase('es-ES')}${rest.join('').toLocaleLowerCase('es-ES')}`;
  }

  function isCompactViewport() {
    return window.matchMedia('(max-width: 900px)').matches;
  }

  function applyDockState(nextCollapsed, persist = true) {
    if (!(searchDock instanceof HTMLElement) || !(dockToggle instanceof HTMLButtonElement)) {
      return;
    }

    const effectiveCollapsed = Boolean(nextCollapsed);
    searchDock.classList.toggle('is-collapsed', effectiveCollapsed);
    dockToggle.setAttribute('aria-expanded', effectiveCollapsed ? 'false' : 'true');
    dockToggle.setAttribute(
      'aria-label',
      effectiveCollapsed
        ? 'Desplegar panel de busqueda'
        : 'Contraer panel de busqueda',
    );
    dockToggle.title = effectiveCollapsed
      ? 'Desplegar panel'
      : 'Contraer panel';
    if (dockContent instanceof HTMLElement) {
      dockContent.setAttribute('aria-hidden', effectiveCollapsed ? 'true' : 'false');
    }

    if (persist) {
      dockCollapsedPreference = Boolean(nextCollapsed);
      try {
        window.localStorage.setItem(DOCK_STORAGE_KEY, nextCollapsed ? '1' : '0');
      } catch {
        // Ignore persistence failures.
      }
    }

    window.dispatchEvent(new CustomEvent('rimasia-dock-resize', {
      detail: { collapsed: effectiveCollapsed },
    }));
  }

  function syncHiddenInput() {
    hiddenInput.value = concepts.join(', ');
  }

  function createChip(value) {
    const chip = document.createElement('span');
    chip.className = 'concept-chip';
    chip.dataset.value = value;

    const label = document.createElement('span');
    label.textContent = toDisplayCapitalized(value);

    const removeButton = document.createElement('button');
    removeButton.type = 'button';
    removeButton.className = 'concept-chip-remove';
    removeButton.setAttribute('aria-label', `Quitar ${toDisplayCapitalized(value)}`);
    removeButton.dataset.removeConcept = value;
    removeButton.textContent = 'x';

    chip.appendChild(label);
    chip.appendChild(removeButton);
    return chip;
  }

  function renderConcepts() {
    const currentChips = chipsContainer.querySelectorAll('.concept-chip');
    currentChips.forEach((chip) => chip.remove());

    const fragment = document.createDocumentFragment();
    concepts.forEach((value) => {
      fragment.appendChild(createChip(value));
    });

    chipsContainer.insertBefore(fragment, input);
    syncHiddenInput();
  }

  function addConceptsFromText(rawText) {
    const candidates = rawText
      .split(',')
      .map((value) => normalize(value))
      .filter(Boolean);

    let changed = false;
    candidates.forEach((value) => {
      if (!concepts.includes(value)) {
        concepts.push(value);
        changed = true;
      }
    });

    if (changed) {
      renderConcepts();
    }
  }

  function commitInputValue() {
    const rawValue = input.value;
    if (!rawValue.trim()) {
      input.value = '';
      return;
    }

    addConceptsFromText(rawValue);
    input.value = '';
  }

  const TOAST_VISIBLE_MS = 5200;
  const TOAST_EXIT_MS = 420;

  function createMessageElement(message, kind) {
    const node = document.createElement('div');
    node.className = `message ${kind === 'warning' ? 'warning-message' : 'error-message'}`;
    node.textContent = message;
    return node;
  }

  function scheduleToastLifecycle(node, delayMs = TOAST_VISIBLE_MS) {
    window.requestAnimationFrame(() => {
      node.classList.add('is-visible');
    });

    window.setTimeout(() => {
      node.classList.add('is-hiding');
      window.setTimeout(() => {
        if (node.parentElement === messageRegion) {
          node.remove();
        }
      }, TOAST_EXIT_MS);
    }, delayMs);
  }

  function hydrateMessageRegion() {
    if (!(messageRegion instanceof HTMLElement)) {
      return;
    }

    Array.from(messageRegion.children).forEach((node, index) => {
      if (!(node instanceof HTMLElement)) {
        return;
      }
      scheduleToastLifecycle(node, TOAST_VISIBLE_MS + index * 240);
    });
  }

  function renderMessages(errorMessage, warningMessage) {
    if (!(messageRegion instanceof HTMLElement)) {
      return;
    }

    messageRegion.textContent = '';

    if (errorMessage) {
      const errorNode = createMessageElement(errorMessage, 'error');
      messageRegion.appendChild(errorNode);
      scheduleToastLifecycle(errorNode);
    }
    if (warningMessage) {
      const warningNode = createMessageElement(warningMessage, 'warning');
      messageRegion.appendChild(warningNode);
      scheduleToastLifecycle(warningNode, TOAST_VISIBLE_MS + 180);
    }
  }

  function syncTargetInputDisplay() {

    if (!(targetInput instanceof HTMLInputElement)) {
      return;
    }

    const trimmedValue = targetInput.value.trim();
    if (!trimmedValue) {
      targetInput.value = '';
      return;
    }

    targetInput.value = toDisplayCapitalized(trimmedValue);
  }

  function setResultPage(nextPage) {
    currentPage = Math.max(0, Number(nextPage) || 0);
    if (resultPageInput instanceof HTMLInputElement) {
      resultPageInput.value = String(currentPage);
    }
  }

  function updateExploreControls(result) {
    if (
      !(exploreControls instanceof HTMLElement)
      || !(exploreMoreButton instanceof HTMLButtonElement)
    ) {
      return;
    }

    currentConsonantTotal = Number(result?.consonant_total || 0);
    currentAssonantTotal = Number(result?.assonant_total || 0);
    currentPageCount = Math.max(1, Number(result?.page_count || 1) || 1);
    setResultPage(result?.result_page || 0);

    exploreControls.dataset.resultPage = String(currentPage);
    exploreControls.dataset.pageCount = String(currentPageCount);
    exploreControls.dataset.consonantTotal = String(currentConsonantTotal);
    exploreControls.dataset.assonantTotal = String(currentAssonantTotal);

    const hasResults = currentConsonantTotal + currentAssonantTotal > 0;
    const hasMoreSamples = hasResults && currentPageCount > 1 && currentPage < currentPageCount - 1;
    exploreControls.classList.toggle('is-hidden', !hasResults);
    exploreMoreButton.classList.toggle('is-hidden', !hasMoreSamples);
    exploreMoreButton.disabled = isLoading || !hasMoreSamples;
    exploreMoreButton.textContent = `DESCUBRIR MÁS (${currentPage + 1} de ${currentPageCount})`;
  }

  function setLoadingState(nextLoading, progress = 0) {
    const submitButton = form?.querySelector('#rhyme-submit');
    if (!(submitButton instanceof HTMLButtonElement)) {
      return;
    }

    isLoading = nextLoading;
    submitButton.classList.toggle('is-loading', nextLoading);
    submitButton.disabled = nextLoading;
    submitButton.setAttribute('aria-busy', nextLoading ? 'true' : 'false');
    loadingTargetProgress = Math.max(0, Math.min(progress, 1));
    if (!nextLoading && loadingTargetProgress === 0) {
      loadingDisplayProgress = 0;
      submitButton.style.setProperty('--load-progress', '0');
    } else if (!loadingRafId) {
      lastLoadingFrame = performance.now();
      loadingRafId = window.requestAnimationFrame(stepLoadingProgress);
    }
    if (exploreMoreButton instanceof HTMLButtonElement) {
      exploreMoreButton.disabled = nextLoading || currentPageCount <= 1;
    }
  }

  function stepLoadingProgress(timestamp) {
    const submitButton = form?.querySelector('#rhyme-submit');
    if (!(submitButton instanceof HTMLButtonElement)) {
      loadingRafId = 0;
      return;
    }

    const deltaMs = Math.min(64, Math.max(16, timestamp - lastLoadingFrame));
    lastLoadingFrame = timestamp;

    const rate = isLoading ? 8.6 : 12.5;
    const blend = 1 - Math.exp(-(rate * deltaMs) / 1000);
    const desiredProgress = isLoading
      ? Math.max(loadingTargetProgress, Math.min(0.985, loadingDisplayProgress + deltaMs * 0.00018))
      : loadingTargetProgress;

    loadingDisplayProgress += (desiredProgress - loadingDisplayProgress) * blend;
    if (!isLoading && loadingTargetProgress === 0 && loadingDisplayProgress < 0.012) {
      loadingDisplayProgress = 0;
    }

    submitButton.style.setProperty('--load-progress', `${Math.max(0, Math.min(loadingDisplayProgress, 1))}`);

    if (isLoading || Math.abs(loadingDisplayProgress - loadingTargetProgress) > 0.004 || loadingTargetProgress > 0) {
      loadingRafId = window.requestAnimationFrame(stepLoadingProgress);
      return;
    }

    loadingRafId = 0;
  }

  function applySearchResult(result) {
    const payload = result?.query_constellation || { nodes: [], edges: [] };
    const highlights = result?.highlight_words || [];

    if (queryScript) {
      queryScript.textContent = JSON.stringify(payload);
    }
    if (highlightScript) {
      highlightScript.textContent = JSON.stringify(highlights);
    }

    if (window.RimasiaConstellation?.applySearchState) {
      window.RimasiaConstellation.applySearchState(payload, highlights);
    }

    updateExploreControls(result);
    renderMessages(null, result?.warning_message || null);
  }

  function wait(ms) {
    return new Promise((resolve) => {
      window.setTimeout(resolve, ms);
    });
  }

  async function pollSearchJob(jobId) {
    while (true) {
      const response = await fetch(`/api/search/status/${jobId}`, {
        headers: { 'X-Requested-With': 'XMLHttpRequest' },
      });
      const payload = await response.json();

      if (!response.ok || !payload.ok) {
        throw new Error(payload.error_message || 'No se pudo seguir el progreso de la consulta.');
      }

      setLoadingState(true, Number(payload.progress || 0));

      if (payload.status === 'complete') {
        setLoadingState(true, 1);
        await wait(180);
        applySearchResult(payload.result || {});
        return;
      }

      if (payload.status === 'error') {
        throw new Error(payload.error_message || 'La consulta ha fallado.');
      }

      await wait(160);
    }
  }

  initialValues.forEach((value) => {
    const normalized = normalize(value);
    if (normalized && !concepts.includes(normalized)) {
      concepts.push(normalized);
    }
  });
  renderConcepts();
  syncTargetInputDisplay();
  applyDockState(dockCollapsedPreference, false);
  updateExploreControls({
    consonant_total: currentConsonantTotal,
    assonant_total: currentAssonantTotal,
    result_page: currentPage,
    page_count: currentPageCount,
    query_constellation: initialQueryPayload,
  });

  hydrateMessageRegion();

  editor.addEventListener('click', () => {
    if (searchDock instanceof HTMLElement && searchDock.classList.contains('is-collapsed')) {
      return;
    }
    input.focus();
  });

  chipsContainer.addEventListener('click', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement) || isLoading) {
      return;
    }

    const conceptToRemove = target.dataset.removeConcept;
    if (!conceptToRemove) {
      return;
    }

    concepts = concepts.filter((value) => value !== conceptToRemove);
    renderConcepts();
    input.focus();
  });

  input.addEventListener('input', () => {
    input.value = toDisplayCapitalized(input.value);
    if (input.value.includes(',')) {
      commitInputValue();
    }
  });

  if (targetInput instanceof HTMLInputElement) {
    targetInput.addEventListener('input', () => {
      const selectionStart = targetInput.selectionStart;
      const selectionEnd = targetInput.selectionEnd;
      const nextValue = toDisplayCapitalized(targetInput.value);
      targetInput.value = nextValue;
      if (selectionStart !== null && selectionEnd !== null) {
        const cursor = nextValue.length;
        targetInput.setSelectionRange(cursor, cursor);
      }
    });

    targetInput.addEventListener('blur', () => {
      syncTargetInputDisplay();
    });
  }

  input.addEventListener('keydown', (event) => {
    if (event.key === ',' || event.key === 'Enter') {
      event.preventDefault();
      commitInputValue();
      return;
    }

    if (event.key === 'Backspace' && !input.value && concepts.length > 0) {
      concepts = concepts.slice(0, -1);
      renderConcepts();
    }
  });

  const form = editor.closest('form');
  async function submitSearch(resetPage, nextPage = null) {
    if (!form || isLoading) {
      return;
    }

    commitInputValue();
    syncHiddenInput();
    const previousPage = currentPage;
    if (resetPage) {
      setResultPage(0);
    } else if (nextPage !== null) {
      setResultPage(nextPage);
    }

    renderMessages(null, null);
    setLoadingState(true, 0.04);

    try {
      const body = new URLSearchParams(new FormData(form));
      const startResponse = await fetch('/api/search/start', {
        method: 'POST',
        headers: { 'X-Requested-With': 'XMLHttpRequest' },
        body,
      });
      const startPayload = await startResponse.json();

      if (!startResponse.ok || !startPayload.ok) {
        throw new Error(startPayload.error_message || 'No se pudo iniciar la consulta.');
      }

      await pollSearchJob(startPayload.job_id);
    } catch (error) {
      setResultPage(previousPage);
      renderMessages(error instanceof Error ? error.message : 'La consulta ha fallado.', null);
    } finally {
      setLoadingState(false, 0);
      updateExploreControls({
        consonant_total: currentConsonantTotal,
        assonant_total: currentAssonantTotal,
        result_page: currentPage,
        page_count: currentPageCount,
      });
    }
  }

  if (form) {
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      await submitSearch(true);
    });
  }

  if (exploreMoreButton instanceof HTMLButtonElement) {
    exploreMoreButton.addEventListener('click', async () => {
      if (isLoading || currentPageCount <= 1 || currentPage >= currentPageCount - 1) {
        return;
      }

      const nextPage = currentPage + 1;
      await submitSearch(false, nextPage);
    });
  }

  if (dockToggle instanceof HTMLButtonElement) {
    dockToggle.addEventListener('click', () => {
      const nextCollapsed = !(searchDock instanceof HTMLElement && searchDock.classList.contains('is-collapsed'));
      applyDockState(nextCollapsed, true);
    });
  }

  window.addEventListener('resize', () => {
    applyDockState(dockCollapsedPreference, false);
  });

  window.addEventListener('pageshow', () => {
    setLoadingState(false, 0);
  });
})();
