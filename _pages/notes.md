---
title: "Notes"
layout: splash
permalink: /notes/
---

<div class="notes-page">
<div class="notes-index">
  <header class="notes-index__header">
    <h1 class="notes-index__title">Notes</h1>
    <p class="notes-index__lead">Some things I think are cool! Recently this looks like deep learning, RL, optimization, and robotics. Click to open the full note.</p>
  </header>

  {% assign note_list = site.pages | where_exp: "p", "p.path == '__no_match__'" %}
  {% for coll in site.collections %}
    {% if coll.label == "notes" %}
      {% assign note_list = coll.docs | sort: "date" | reverse %}
    {% endif %}
  {% endfor %}

  <div class="notes-index__list">
    {% for note in note_list %}
      <article class="notes-card">
        <a class="notes-card__link" href="{{ note.url | relative_url }}" aria-label="Open note: {{ note.title | strip | escape }}"></a>
        <div class="notes-card__inner">
          <h2 class="notes-card__title">{{ note.title }}</h2>
          {% assign blurb = note.description | default: note.excerpt %}
          {% if blurb %}
            <p class="notes-card__desc">{{ blurb | strip_html | truncate: 200 }}</p>
          {% endif %}
          <div class="notes-card__meta">
            {% if note.date %}
              <time class="notes-card__date" datetime="{{ note.date | date_to_xmlschema }}">{{ note.date | date: "%b %d, %Y" }}</time>
            {% endif %}
            {% if note.tag %}
              <span class="notes-card__hashtag">#{{ note.tag }}</span>
            {% endif %}
          </div>
        </div>
      </article>
    {% endfor %}
  </div>
</div>

<h3 class="notes-page__subhead" id="draft-topics">Draft topics</h3>
<ul class="notes-page__list">
  <li><em>TBD</em></li>
</ul>
<p class="notes-page__footnote">Thanks for reading!</p>

</div>
