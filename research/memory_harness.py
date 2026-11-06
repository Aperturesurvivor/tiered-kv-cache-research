"""Bounded, fallible model notes and KV-address history; never evaluator data.

The state is reset at episode boundaries. All updates consume only public
questions, model output and the addresses actually selected by the retriever.
Token limits apply to the pinned generator tokenizer, not whitespace words.
"""
from dataclasses import asdict, dataclass
import re

from research.reasoning_protocol import parse_response


MODES = ('none', 'notes', 'addresses', 'both')
MAX_ENTRIES = 4
STATE_TOKEN_CAP = 256
PROMPT_TOKEN_CAP = 256
ADDRESS_CAP = 4  # per entry; at most 16 address integers in RAM
NOTICE = ('Unverified memory from earlier requests follows. It may be wrong or '
          'outdated. Check it against the archive and the current question.\n')


def clipped(text, tok, cap):
    ids = tok.encode(text, add_special_tokens=False)[:cap]
    value = tok.decode(ids, skip_special_tokens=True)
    # Some token boundaries decode through a replacement character. Bound the
    # actual re-encoded input as well as its initially sliced representation.
    while len(tok.encode(value, add_special_tokens=False)) > cap:
        ids = ids[:-1]
        value = tok.decode(ids, skip_special_tokens=True)
    return value


@dataclass
class Entry:
    question: str
    answer: str
    search: str
    notes: str
    blocks: tuple


class BoundedMemory:
    def __init__(self, tok, mode):
        if mode not in MODES:
            raise ValueError('unknown memory mode')
        self.tok, self.mode, self.entries = tok, mode, []

    def token_count(self):
        return sum(len(self.tok.encode(value, add_special_tokens=False))
                   for e in self.entries for value in
                   (e.question, e.answer, e.search, e.notes))

    def snapshot(self):
        return {'mode': self.mode, 'entries': [{**asdict(e), 'blocks':list(e.blocks)} for e in self.entries],
                'stored_text_tokens': self.token_count(),
                'stored_addresses': sum(len(e.blocks) for e in self.entries)}

    def restore(self, snapshot):
        if snapshot['mode'] != self.mode:
            raise ValueError('memory mode differs')
        entries = [Entry(**{**e, 'blocks':tuple(e['blocks'])}) for e in snapshot['entries']]
        previous = self.entries
        self.entries = entries
        try:
            if (len(entries) > MAX_ENTRIES or self.token_count() > STATE_TOKEN_CAP
                    or any(len(e.blocks) > ADDRESS_CAP or
                           any(type(b) is not int or b < 0 for b in e.blocks) for e in entries)
                    or self.snapshot() != snapshot or (self.mode == 'none' and entries)):
                raise ValueError('invalid bounded memory snapshot')
        except BaseException:
            self.entries = previous
            raise

    def observe(self, question, response, selected_blocks):
        """Invalid JSON is not repaired or saved; a valid wrong answer is saved."""
        parsed = parse_response(response)
        if self.mode == 'none' or parsed is None:
            return {'saved': False, 'valid_response': parsed is not None, 'evicted': 0}
        blocks = list(dict.fromkeys(selected_blocks))[:ADDRESS_CAP]
        if any(type(b) is not int or b < 0 for b in blocks):
            raise ValueError('nonnegative integer addresses required')
        entry = Entry(clipped(question, self.tok, 32),
                      clipped(parsed['answer'], self.tok, 16),
                      clipped(parsed['search'], self.tok, 32),
                      clipped(parsed['notes'], self.tok, 48), tuple(blocks))
        self.entries.append(entry)
        evicted = 0
        while len(self.entries) > MAX_ENTRIES or self.token_count() > STATE_TOKEN_CAP:
            self.entries.pop(0)
            evicted += 1
        return {'saved': True, 'valid_response': True, 'evicted': evicted}

    def relevant(self, question):
        # Bounded linear work over <=4 entries. Match public question terms;
        # never inspect the archive, evaluator answer, or target record spans.
        terms = lambda s: set(re.findall(r'[a-z0-9_]+', s.lower())) - {
            'a','an','the','what','which','is','was','are','for','of','in','at',
            'to','and','or','please','answer','code','access','current','now'}
        query = terms(question)
        scores = [(len(query & terms(e.question)), i, e)
                  for i, e in enumerate(self.entries)]
        match = max(scores, key=lambda v: (v[0], v[1]), default=(0, -1, None))
        return match[2] if match[0] else None

    def prompt(self):
        if self.mode not in ('notes', 'both') or not self.entries:
            return ''
        text = NOTICE + '\n'.join(
            f'Question: {e.question}\nModel answer: {e.answer}\nModel notes: {e.notes}'
            for e in reversed(self.entries))
        return clipped(text, self.tok, PROMPT_TOKEN_CAP)

    def addresses(self, question, eligible):
        entry = self.relevant(question) if self.mode in ('addresses','both') else None
        return [b for b in entry.blocks if b < eligible] if entry else []
